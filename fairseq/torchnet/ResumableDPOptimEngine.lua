-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A version of OptimEngine that implements data parallelism and
-- can resume itself.
--
--]]

local tnt = require 'torchnet'
local argcheck = require 'argcheck'
local utils = require 'fairseq.utils'
local threads = require 'threads'
local mutils = require 'fairseq.models.utils'

local cuda = utils.loadCuda()

local ResumableDPOptimEngine =
    torch.class('tnt.ResumableDPOptimEngine', 'tnt.OptimEngine', tnt)

function transfer(src, tgt)
    sp, sgp = src:parameters()
    tp, tgp = tgt:parameters()
    for i = 1, #tp do
        tp[i]:copy(sp[i])
    end
end

ResumableDPOptimEngine.__init = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='nshards', type='number'},
    {name='thread_init_fn', type='function'},
    {name='make_model_fn', type='function'},
    {name='make_criterion_fn', type='function'},
    {name='config', type='table'},
    call = function(self, nshards, thread_init_fn,
        make_model_fn, make_criterion_fn, config)
        tnt.Engine.__init(self, {
            'onStart', 'onStartEpoch', 'onSample',
            'onForward', 'onForwardCriterion',
            'onBackward', 'onBackwardCriterion',
            'onEndEpoch', 'onUpdate', 'onEnd',
            'onResume', 'onJumpToEpoch', 'onJumpToSample',
            'onEmbedding',
        })

        self.nshards = nshards
        threads.Threads.serialization('threads.sharedserialize')
        self.pool = threads.Threads(
            nshards,
            thread_init_fn,
            function(id)
               
                local cutorch = require 'cutorch'
                cutorch.setDevice(id)
                _G.id = id
                _G.model = make_model_fn(id)
                _G.params, _G.gradparams = _G.model:network():getParameters()

                if not config.pretrain then
                    print('start transfer parameters from an old model')
                    input_model = torch.load(config.pretrainpath .. '/model_best.th7')
                    input_model.module:clearState()

                    _G.model.module:clearState()

                    p_G, _ = _G.model.module:getParameters()
                    p_i, _ = input_model.module:getParameters()
                    p_G:copy(p_i)
                end
                
                _G.criterion, _G.critweights = make_criterion_fn(id)
                _G.optstate = {}
                _G.feval = function()
                    return _G.criterion.output, _G.gradparams
                end
                _G.clipgrads = function(clipv)
                    if clipv > 0 then
                        local norm = _G.gradparams:norm()
                        if norm > clipv then
                            local coef = math.max(norm, 1e-6) / clipv
                            _G.gradparams:div(coef)
                        end
                    end
                end
                _G.prepareSample = _G.model:prepareSample()
                _G.model:network():zeroGradParameters()
                if _G.criterion.zeroGradParameters then
                    _G.criterion:zeroGradParameters()
                end
            end
        )
        self.pool:specific(true)
    end
}

ResumableDPOptimEngine.model = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    call = function(self)
        local model = nil
        self.pool:addjob(1,
            function() return _G.model end,
            function(m) model = m end
        )
        self.pool:synchronize()
        return model
    end
}

ResumableDPOptimEngine.saveState = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='path', type='string'},
    {name='state', type='table'},
    call = function(self, path, state)
        local save = {
            t = state.t,
            epoch_t = state.epoch_t,
            epoch = state.epoch,
            optconfig = state.optconfig,
            clip = state.clip,
            maxepoch = state.maxepoch,
        }
        self.pool:addjob(1,
            function(path, save)
                local utils = require 'fairseq.utils'
                save.params = _G.params
                save.optstate = _G.optstate
                if utils.retry(3, torch.save, path .. '.t', save) then
                    utils.retry(3, os.rename, path .. '.t', path)
                end
            end,
            function() end,
            path, save
        )
        self.pool:synchronize()
    end
}

ResumableDPOptimEngine.loadState = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='path', type='string'},
    call = function(self, path)
        print('load state...', path)
        local state = torch.load(path)
        for shardid = 1, self.nshards do
            self.pool:addjob(shardid,
                function(params, optstate)
                    assert(_G.params:nElement() == params:nElement())
                    _G.params:copy(params)
                    _G.optstate = {}
                    for k, v in pairs(optstate) do
                        if torch.type(v) == 'torch.CudaTensor' then
                            _G.optstate[k] = v.new(v:size()):copy(v)
                        else
                            _G.optstate[k] = v
                        end
                    end
                end,
                function() end,
                state.params, state.optstate
            )
        end
        self.pool:synchronize()
        state.params = nil
        state.optstate = nil
        collectgarbage()
        state.epoch_t = state.epoch_t or 0
        state.rng_state = state.rng_state or torch.getRNGState()
        state.cu_rng_state = state.cu_rng_state or cuda.cutorch.getRNGState()
        state.optconfig.timeAverage = state.optconfig.timeAverage
            or state.optconfig.sizeAverage
        return state
    end
}

ResumableDPOptimEngine.resume = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='path', type='string'},
    {name='iterator', type='tnt.DatasetIterator'},
    {name='config', type='table'},
    call = function(self, path, iterator, config)
        local state = self:loadState(path)
        state.iterator = iterator
        self.hooks('onResume', state)
        self:doTrain(state, config)
    end
}

ResumableDPOptimEngine.train = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='iterator', type='tnt.DatasetIterator'},
    {name='optconfig', type='table'},
    {name='config', type='table'},
    {name='maxepoch', type='number', default=1000},
    {name='clip', type='number', default=10},
    call = function(self, iterator, optconfig, config, maxepoch, clip)
        local state = {
            iterator = iterator,
            maxepoch = maxepoch,
            clip = clip,
            optconfig = optconfig,
            epoch = 0, 
            t = 0, 
            epoch_t = 0, 
            params = {},
            gradparams = {},
        }
        self:doTrain(state, config)
    end
}

ResumableDPOptimEngine.test = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='iterator', type='tnt.DatasetIterator'},
    call = function(self, iterator)
        local state = {
            iterator = iterator,
        }
        self:evaluate()
        for samples in state.iterator() do
            state.samples = samples
            self.hooks('onSample', state)

            state.loss = 0
            state.ntokens = 0
            for shardid = 1, self.nshards do
                local sample = state.samples[shardid]
                if sample then
                    state.ntokens = state.ntokens + sample.ntokens
                    self.pool:addjob(shardid,
                        function(sample)
                            _G.prepareSample(sample)
                            _G.model:resizeCriterionWeights(
                                _G.criterion, _G.critweights, sample)
                            local net = _G.model:network()
                            local crit = _G.criterion
                            net:forward(sample.input)
                            crit:forward(net.output, sample.target)
                            collectgarbage()
                            return crit.output
                        end,
                        function(loss)
                            state.loss = state.loss + loss
                        end,
                        sample
                    )
                end
            end
            self.pool:synchronize()
            self.hooks('onForwardCriterion', state)
        end
    end
}

ResumableDPOptimEngine.modifyEmbedding = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='state', type='table'},
    {name='config', type='table'},
    call = function(self, state, config)
    
    end
}

ResumableDPOptimEngine.modifyEmbedding2 = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='state', type='table'},
    {name='config', type='table'},
    call = function(self, state, config)


    end
}

ResumableDPOptimEngine.updateFrequency = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='samples', type='table'},
    {name='config', type='table'},
    call = function(self, samples, config)
        local timer = torch.Timer()
        
    end
}

ResumableDPOptimEngine.accFrequency = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='samples', type='table'},
    {name='config', type='table'},
    call = function(self, samples, config)
        local timer = torch.Timer()
        
    end
}

ResumableDPOptimEngine.executeAll = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='fn', type='function'},
    {name='collect_fn', type='function', default=function() end},
    call = function(self, fn, collect_fn)
        for shardid = 1, self.nshards do
            self.pool:addjob(shardid, fn, collect_fn, shardid)
        end
        self.pool:synchronize()
    end
}

ResumableDPOptimEngine.saveModel = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='modelpath', type='string'},
    call = function(self, modelpath)
        self.pool:addjob(1,
            function()
                _G.model:network():clearState()
                torch.save(modelpath, _G.model)
            end
        )
        self.pool:synchronize()
    end
}

ResumableDPOptimEngine.training = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    call = function(self)
        self:executeAll(function(id) _G.model:network():training() end)
    end
}

ResumableDPOptimEngine.evaluate = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    call = function(self)
        self:executeAll(function(id) _G.model:network():evaluate() end)
    end
}

ResumableDPOptimEngine.doTrain = argcheck{
    {name='self', type='tnt.ResumableDPOptimEngine'},
    {name='state', type='table'},
    {name='config', type='table'},
    call = function(self, state, config)
        state.params = {}
        state.gradparams = {}
        for shardid = 1, self.nshards do
            self.pool:addjob(shardid,
                function()
                    return _G.params, _G.gradparams
                end,
                function(p, gp)
                    state.params[shardid] = p
                    state.gradparams[shardid] = gp
                end
            )
        end
        self.pool:synchronize()
        -- Synchronize weights before training begins
        cuda.nccl.bcast(state.params, false, 1)

        local calcclipv = function(samples, timeAverage)
            local clipv = 0
            if not timeAverage then
                for _, sample in ipairs(samples) do
                    clipv = clipv + sample.bsz
                end
            else
                clipv = 1
            end
            return clipv * state.clip
        end

        state.jumped = 0
        local jumping = state.jumped < state.t

        self.hooks('onStart', state)
        print('calling onStart function from hook')
        while state.epoch < state.maxepoch do
            self:training()

            jumping = jumping and state.jumped < state.epoch_t
            if not jumping then
                self.hooks('onStartEpoch', state)
            else
                self.hooks('onJumpToEpoch', state)
            end
            local prevn = 1
            local cntiter = 0
            local clipv = state.clip
            for samples in state.iterator() do
                jumping = jumping and state.jumped < state.epoch_t
                if not jumping then
                    state.samples = samples
                    self.hooks('onSample', state)

                    state.loss = 0
                    state.ntokens = 0
                    for shardid = 1, self.nshards do
                        local sample = state.samples[shardid]
                        if sample then
                            state.ntokens = state.ntokens + sample.ntokens
                            
                            self.pool:addjob(shardid,
                                function(optconfig, sample, clipv, prevn)
                                    if optconfig then
                                        if optconfig.timeAverage then
                                            _G.gradparams:div(prevn)
                                        end
                                        _G.clipgrads(clipv)
                                        optconfig.method(_G.feval, _G.params,
                                            optconfig, _G.optstate)
                                    end

                                    -- Process the current sample.
                                    _G.prepareSample(sample)
                                    _G.model:resizeCriterionWeights(
                                        _G.criterion, _G.critweights, sample)
                                    local net = _G.model:network()
                                    local crit = _G.criterion
                                    net:zeroGradParameters()
                                    if crit.zeroGradParameters then
                                        crit:zeroGradParameters()
                                    end
                                    net:forward(sample.input)
                                    crit:forward(net.output, sample.target)
                                    crit:backward(net.output, sample.target)
                                    net:backward(sample.input, crit.gradInput)

                                    collectgarbage()
                                    return crit.output
                                end,
                                function(loss)
                                    state.loss = state.loss + loss
                                end,
                                state.epoch_t > 0 and state.optconfig or nil,
                                    sample, clipv, prevn
                            )
                        end
                    end
                    self.pool:synchronize()
                    cuda.nccl.allReduce(state.gradparams, state.gradparams, true)

                    clipv = calcclipv(samples, state.optconfig.timeAverage)
                    prevn = state.ntokens
                    state.t = state.t + 1
                    state.epoch_t = state.epoch_t + 1
                    self.hooks('onUpdate', state)
                else
                    state.jumped = state.jumped + 1
                    self.hooks('onJumpToSample', state)
                end
            end

            -- End of epoch: perform the last optimization step
            if not jumping then
                for shardid = 1, self.nshards do
                    self.pool:addjob(shardid,
                        function(optconfig, clipv, prevn)
                            if optconfig then
                                if optconfig.timeAverage then
                                    _G.gradparams:div(prevn)
                                end
                                _G.clipgrads(clipv)
                                optconfig.method(_G.feval, _G.params,
                                    optconfig, _G.optstate)
                            end
                        end,
                        function() end,
                        state.epoch_t > 0 and state.optconfig or nil,
                            clipv, prevn
                    )
                    self.pool:synchronize()
                end
            end

            state.epoch = state.epoch + 1
            state.epoch_t = 0
            self.hooks('onEndEpoch', state)
        end
        self.hooks('onEnd', state)
    end
}
