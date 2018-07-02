-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- A fully convolutional model, i.e. a convolutional encoder and a convolutional
-- decoder language model. This model supports vocabulary selection and will
-- construct an appropriate output layer if config.aligndict is set.
--
--]]

require 'nn'
require 'nngraph'
require 'rnnlib'
require 'fairseq.modules'
local pltablex = require 'pl.tablex'
local plutils = require 'pl.utils'
local argcheck = require 'argcheck'
local tds = require 'tds'
local mutils = require 'fairseq.models.utils'
local utils = require 'fairseq.utils'

local cuda = utils.loadCuda()

local function attentionLayers(nlayer, attnlayers)
    local attnlayersTab = tds.Hash(pltablex.index_map(
        pltablex.map(function(n) return tonumber(n) end,
                      plutils.split(attnlayers, ','))
    ))
    -- -1 = attn everywhere
    if attnlayersTab[-1] then
        for i = 1, nlayer do
            attnlayersTab[i] = i
        end
    end
    attnlayersTab[-1] = nil
    for i = nlayer + 1, #attnlayersTab do
        attnlayersTab[i] = nil
    end
    return attnlayersTab
end
local function calHidNum(config)
    local encIn, encOut, decIn, decOut = {}, {}, {}, {}
    encIn[1] = config.nhids[1]
    if config.optenc == 0 or 1 <= config.keeplayers_enc then
        encOut[1] = config.nhids[1]
    elseif config.optenc == 1 then
        encOut[1] = config.nhids[1] + config.nhid_accs[1]
    end
    decIn[1] = config.nlmhids[1]
    if config.opt == 0 or 1 <= config.keeplayers_dec then
        decOut[1] = config.nhids[1]
    elseif config.opt == 1 then
        decOut[1] = config.nhids[1] + 2* config.nhid_accs_dec[1]
    elseif config.opt == 2 then
        decOut[1] = config.nhids[1] + config.nhid_accs_dec[1]
    elseif config.opt == 3 or config.opt == 4 then
        decOut[1] = 2*config.nhid_accs_dec[1]
    else 
        print("Not Implemented Error")
    end

    for i = 2, #config.nhids do
        encIn[i] = encOut[i-1]
        if config.optenc == 0 or i <= config.keeplayers_enc then
            encOut[i] = config.nhids[i]
        elseif config.optenc == 1 then
            if (i-config.keeplayers_enc) % config.blocklength == 0 then
                encOut[i] = config.nhids[i]
            else
                encOut[i] = encIn[i] + config.nhid_accs[i]
            end
        else
            print("Not Implemented Error")
        end
    end
    for i = 2, #config.nlmhids do
        decIn[i] = decOut[i-1]
        if config.opt == 0 or i <= config.keeplayers_dec then
            decOut[i] = config.nlmhids[i]
        elseif config.opt == 1 or config.opt == 2 then
            if (i-config.keeplayers_dec) % config.blocklength == 0 then
                decOut[i] = config.nlmhids[i]
            else
                decOut[i] = decIn[i] + (3 - config.opt) * config.nhid_accs_dec[i]
            end
        elseif config.opt == 3 or config.opt == 4 then
            if (i-config.keeplayers_dec) % config.blocklength == 0 then
                decOut[i] = config.nlmhids[i]
            else
                decOut[i] = decIn[i] + 2 * config.nhid_accs_dec[i]
            end
        else
            print("Not Implemented Error")
        end
    end
    res = {}
    res['encIn'] = encIn
    res['encOut'] = encOut
    res['decIn'] = decIn
    res['decOut'] = decOut
    return res
end
local FConvDenseMoptModel, parent = torch.class('FConvDenseMoptModel', 'AvgpoolModel')

FConvDenseMoptModel.__init = argcheck{
    {name='self', type='Model'},
    {name='config', type='table', opt=true},
    call = function(self, config)
        parent.__init(self, config)
        config.numhid = calHidNum(config)
        print(config.numhid)
        -- Store maximum context accessible by the decoder so that generation
        -- can skip unnecessary computations.
        local decoder = mutils.findAnnotatedNode(self:network(), 'decoder')
        local context = 1
        decoder:apply(function(m)
            if torch.isTypeOf(m, 'cudnn.TemporalConvolution') then
                context = context + m.kH - 1
            elseif torch.isTypeOf(m, 'nn.TemporalConvolutionTBC') then
                context = context + m.kw - 1
            end
        end)
        self.decoderContext = context
    end
}

local ConvDenseMoptBlockTrainTestLayer, _ = torch.class('nn.ConvDenseMoptBlockTrainTestLayer',
    'nn.TrainTestLayer')

function ConvDenseMoptBlockTrainTestLayer:__init(trainModule, evalModule)
    nn.Container.__init(self)
    self.modules[1] = trainModule
    self.modules[2] = evalModule
    self.train = true
end

ConvDenseMoptBlockTrainTestLayer.onTrain = function() end
ConvDenseMoptBlockTrainTestLayer.onEvaluate = function(trnMod, testMod)
    local conv = trnMod:get(2):get(1)
    testMod:setParameters(
        conv.weight:transpose(3, 1):transpose(2, 3),
        conv.bias
    )
end





FConvDenseMoptModel.makeDecoderFast = argcheck{
    {name='self', type='Model'},
    call = function(self)
        if self.convStates then return end

        -- edit the conv blocks to use LinearizedConvolution at test time
        local decoder = mutils.findAnnotatedNode(self:network(), 'decoder')
        for i = 1, math.huge do
            local block = mutils.findAnnotatedNode(decoder, 'ConvDenseMoptBlock_' .. i)
            if not block then break end

            -- typecheck
            local mismatch = 'type mismatch'
            assert(block:size() == 5, mismatch)
            local layertypes = {
                'nn.Transpose', 'nn.WeightNorm', 'nn.Transpose', 'nn.Narrow',
                'nn.GatedLinearUnit'
            }
            for i = 1, 5 do
                assert(torch.isTypeOf(block:get(i), layertypes[i]), mismatch)
            end
            local conv = block:get(2):get(1)
            assert(torch.isTypeOf(conv, 'nn.TemporalConvolutionTBC'), mismatch)

            -- build a layer to allow different path for train and decode
            local trnMod = nn.Sequential()
                :add(block:get(1))
                :add(block:get(2))
                :add(block:get(3))
                :add(block:get(4))
            local testMod = nn.LinearizedConvolution()

            -- new layers match original type
            trnMod:type(conv:type())
            testMod:type(conv:type())
            local trnTest = nn.ConvDenseMoptBlockTrainTestLayer(trnMod, testMod)
            trnTest:type(conv:type())

            -- replace first four layer by train/test layer
            for i = 1, 4 do block:remove(1) end
            block:insert(trnTest, 1)
            self.convStates = self.convStates or {}
            table.insert(self.convStates, testMod)
        end

        -- edit the attention modules to use beamableMM (for old models)
        decoder:apply(function(m)
            if torch.typename(m) == 'nn.MM' then
                return nn.BeamableMM()
            end
        end)
    end
}

FConvDenseMoptModel.setBeamSize = argcheck{
    {name='self', type='Model'},
    {name='beam', type='number'},
    call = function(self, beam)
        if self.beamSize ~= beam then -- beam size has changed?
            self:network():apply(function(m) -- change it for all layers
                if m.setBeamSize then -- that support setting beam size
                    m:setBeamSize(beam)
                end
            end)
            self.beamSize = beam
        end
    end
}

FConvDenseMoptModel.makeLookupTable = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='nindex', type='number'},
    {name='noutput', type='number'},
    {name='paddingValue', type='number', opt=true},
    call = function(self, config, nindex, noutput, paddingValue)
        local lut = nn.LookupTable(nindex, noutput, paddingValue)
        lut.weight:normal(0, 0.1)
        return lut
    end
}

FConvDenseMoptModel.makeEmbedding = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='dict', type='Dictionary'},
    {name='dropout', type='number', opt=true},
    call = function(self, config, dict, dropout)
        local maxPosition = 1024
        local embedTokens = self:makeModuleName(
            self:makeLookupTable(config, dict:size(), config.ninembed, dict:getPadIndex()),
            'WordEmbed')
        
        local embedPositions = self:makeModuleName(
            self:makeLookupTable(config, maxPosition, config.ninembed, dict:getPadIndex()),
            'PosEmbed')

        -- Expected input: {tokens, positions}

        local embed = nn.Sequential()
        if config.ninembed ~= config.nembed then
            embed:add(nn.ParallelTable()
                :add(nn.Sequential()
                    :add(embedTokens)
                    :add(nn.Bottle(self:makeLinear(config.ninembed, config.nembed, config.dropout_src))) )
                :add(nn.Sequential()
                    :add(embedPositions)
                    :add(nn.Bottle(self:makeLinear(config.ninembed, config.nembed, config.dropout_src))) )
            )
        else
            embed:add(nn.ParallelTable()
            :add(embedTokens)
            :add(embedPositions)
            )
        end
        embed:add(nn.CAddTable())
        embed:add(self:makeDropout(dropout))
        return embed
    end
}

FConvDenseMoptModel.makeTemporalConv = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='cudnnconv', type='boolean'},
    {name='ninput', type='number'},
    {name='noutput', type='number'},
    {name='kwidth', type='number'},
    {name='pad', type='number'},
    {name='doin', type='number', default=0},
    call = function(self, cudnnconv, ninput, noutput, kwidth, pad, doin)
        local conv
        if cudnnconv then
            conv = nn.WeightNorm(cuda.cudnn.TemporalConvolution(
                ninput, noutput, kwidth, 1, pad))
        else
            conv = nn.WeightNorm(
                nn.TemporalConvolutionTBC(ninput, noutput, kwidth, pad),
                3) --outputDim
        end
        conv.v:normal(0, math.sqrt(4 * (1 - doin) / (kwidth * ninput)))
        conv.g:norm(conv.v, 2, 2)
        return conv
    end
}

FConvDenseMoptModel.makeBottleneckConv = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='cudnnconv', type='boolean'},
    {name='ninput', type='number'},
    {name='noutput', type='number'},
    {name='bnhid', type='number'},
    {name='kwidth', type='number'},
    {name='pad', type='number'},
    {name='doin', type='number', default=0},
    call = function(self, cudnnconv, ninput, noutput, bnhid, kwidth, pad,
      doin)
        local bottleneck = nn.Sequential()
        bottleneck:add(nn.Bottle(self:makeLinear(ninput, bnhid*2, doin)))
        bottleneck:add(nn.GatedLinearUnit())
        bottleneck:add(self:makeTemporalConv(
            cudnnconv, bnhid, bnhid*2, kwidth, pad)
        )
        bottleneck:add(nn.GatedLinearUnit())
        bottleneck:add(nn.Bottle(self:makeLinear(bnhid, noutput)))
        -- bottleneck:add(nn.GatedLinearUnit())
        return bottleneck
    end
}

FConvDenseMoptModel.makeTranspose = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='cudnnconv', type='boolean'},
    call = function(self, cudnnconv)
        if cudnnconv then
            return nn.Identity()
        else
            return nn.Transpose({1, 2})
        end
    end
}

FConvDenseMoptModel.makeEncoderConvBlock = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='cudnnconv', type='boolean'},
    {name='ninput', type='number'},
    {name='noutput', type='number'},
    {name='bfactor', type='number'},
    {name='kwidth', type='number'},
    {name='doin', type='number', default=0},
    call = function(self, cudnnconv, ninput, noutput, bfactor, kwidth, doin)
        local pad = (kwidth - 1) / 2
        local block = nn.Sequential()
        local conv
        if bfactor > 0 then
            local bnhid = ninput / bfactor
            conv = self:makeBottleneckConv(
                cudnnconv, ninput, noutput * 2, bnhid, kwidth, pad, doin)
        else
            conv = self:makeTemporalConv(
                cudnnconv, ninput, noutput * 2, kwidth, pad, doin)
        end
        block:add(conv)
        block:add(nn.GatedLinearUnit())
        return block
    end
}

FConvDenseMoptModel.makeDecoderConvBlock = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='cudnnconv', type='boolean'},
    {name='ninput', type='number'},
    {name='noutput', type='number'},
    {name='bfactor', type='number'},
    {name='kwidth', type='number'},
    {name='doin', type='number', default=0},
    call = function(self, cudnnconv, ninput, noutput, bfactor, kwidth, doin)
        local pad = kwidth - 1
        local block = nn.Sequential()
        local conv
        if bfactor > 0 then
            local bnhid = ninput / bfactor
            conv = self:makeBottleneckConv(
                cudnnconv, ninput, noutput * 2, bnhid, kwidth, pad, doin)
        else
            conv = self:makeTemporalConv(
                cudnnconv, ninput, noutput * 2, kwidth, pad, doin)
        end
        block:add(self:makeTranspose(cudnnconv))
        block:add(conv)
        block:add(self:makeTranspose(cudnnconv))
        -- remove future timestamps
        block:add(nn.Narrow(2, 1, -kwidth))
        block:add(nn.GatedLinearUnit())
        return block
    end
}

FConvDenseMoptModel.makeLinear = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='ninput', type='number'},
    {name='noutput', type='number'},
    {name='doin', type='number', default=0},
    call = function(self, ninput, noutput, doin)
        local m = nn.WeightNorm(nn.Linear(ninput, noutput))
        m.v:normal(0, math.sqrt((1-doin) / ninput))
        m.g:norm(m.v, 2, 2)
        return m
    end
}

FConvDenseMoptModel.makeDropout = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='p', type='number', opt=true},
    call = function(self, p)
        if not p or p == 0 then
            return nn.Identity()
        end
        return nn.Dropout(p)
    end
}

FConvDenseMoptModel.makeModuleName = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='module', type='nn.Module'},
    {name='name', type='string'},
    call = function(self, module, name)
        module.name = name
        return module 
    end
}

FConvDenseMoptModel.makeEncoderStack = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        if #config.nhids == 0 then
            return nn.Identity()
        end

        local stack = nn.Sequential()
        stack:add(self:makeTranspose(config.cudnnconv))
        config.numhid = calHidNum(config)
        --local nhid_acc = 256
        for i = 1, #config.nhids do
            local nhidin = config.nhids[i - 1] or config.nhids[i]
            local nhidout = config.nhids[i]

            local nhidin_real = config.numhid.encIn[i]
            local nhidout_real = config.numhid.encOut[i]

            --[[
            if config.optenc == 0 then
                nhidin_real = nhidin
                nhidout_real = nhidout
            elseif config.optenc == 1 then
                nhidin_real = nhidin + ((i-1)%config.blocklength) * config.nhid_acc
                nhidout_real = nhidout + ((i-1)%config.blocklength+1) * config.nhid_acc
            end
            --]]


            local inpmap = nn.Identity()

            --
            if (config.optenc == 0 and nhidin ~= nhidout)  then
                
                inpmap = self:makeModuleName(
                    nn.Bottle(
                        self:makeLinear(nhidin, nhidout, config.dropout_hid)),
                    'encoderConv_inpmap_' .. i)
            
            end
            if  (i-config.keeplayers_enc) % config.blocklength == 0  then
                inpmap = self:makeModuleName(
                    nn.Bottle(
                        self:makeLinear(nhidin_real, nhidout, config.dropout_hid)),
                    'encoderConv_inpmap_' .. i)
            
            end
            --]]

            local convout 
            if config.optenc == 0 or (i-config.keeplayers_enc) % config.blocklength == 0  then
                convout = nhidout_real
            elseif config.optenc == 1 then
                convout = config.nhid_accs[i]
            end

            -- Residual connections
            if config.optenc == 0 or (i-config.keeplayers_enc) % config.blocklength == 0 then
                stack:add(nn.ConcatTable()
                    :add(nn.Sequential()
                        :add(self:makeDropout(config.dropout_hid))
                        :add(self:makeModuleName(
                                self:makeEncoderConvBlock(config.cudnnconv,
                                nhidin_real, convout, config.bfactor, config.kwidths[i],
                                config.dropout_hid),
                                'encoderConv_' .. i)))
                    :add(inpmap))
                stack:add(nn.CAddTableMulConstant(math.sqrt(0.5)))
            elseif config.optenc == 1 then
                stack:add(nn.Concat(3)
                    :add(nn.Sequential()
                        :add(self:makeDropout(config.dropout_hid))
                        :add(self:makeModuleName(
                                self:makeEncoderConvBlock(config.cudnnconv,
                                nhidin_real, convout, config.bfactor, config.kwidths[i],
                                config.dropout_hid),
                                'encoderConv_' .. i)))
                    :add(inpmap))
            end
        end
        stack:add(self:makeTranspose(config.cudnnconv))
        return stack
    end
}

FConvDenseMoptModel.makeEncoder = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        local sourceIn = nn.Identity()()
        local tokens, positions = sourceIn:split(2)

        local embedmodel = self:makeEmbedding(config, config.srcdict, config.dropout_src)
        --self.embed = embedmodel
        local embed = embedmodel({tokens, positions}):annotate({name = 'encoderEmbed'})
        --local embed = self:makeEmbedding(config, config.srcdict,
        --    config.dropout_src)({tokens, positions}):annotate({name = 'encoderEmbed'})
        
        config.numhid = calHidNum(config)

        local cnn
        if #config.nhids > 0 then
            cnn = nn.Sequential()
            cnn:add(self:makeModuleName(
                nn.Bottle(
                    self:makeLinear(config.nembed, config.nhids[1],
                    config.dropout_src)),
                'encoderEmbedToCNN'))
            cnn:add(nn.Contiguous())
            cnn:add(self:makeEncoderStack(config))
            
            local nhidout_real = config.numhid.encOut[#config.nhids]
            --[[
            if config.optenc == 0 or #config.nhids%config.blocklength == 0 then
                nhidout_real = config.nhids[#config.nhids]
            elseif config.optenc == 1 then
                nhidout_real = ((#config.nhids-1)%config.blocklength+1) * config.nhid_acc + config.nhids[#config.nhids]
            end
            --]]
            
            cnn:add(self:makeModuleName(
                nn.Bottle(
                    self:makeLinear(nhidout_real, config.nembed)),
                'encoderCNNToEmbed'))
        else
            cnn = nn.Identity()
        end

        -- The encoder stack will receive gradients *twice* for each attention
        -- pass: dot product and weighted sum.
        local nattn = #attentionLayers(#config.nlmhids, config.attnlayers)
        cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))

        local outputA = cnn(embed):annotate({name = 'encoderCNN'})
        -- Source feeding for weighted sum in attention: add embeddings to CNN
        -- output
        local outputC = nn.CAddTableMulConstant(
            math.sqrt(0.5))({outputA, embed})
        return nn.gModule({sourceIn}, {outputA, outputC})
    end
}

FConvDenseMoptModel.makeEncoderDense = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        local sourceIn = nn.Identity()()
        local tokens, positions = sourceIn:split(2)

        local embedmodel = self:makeEmbedding(config, config.srcdict, config.dropout_src)
        --self.embed = embedmodel
        local embed = embedmodel({tokens, positions}):annotate({name = 'encoderEmbed'})
        --local embed = self:makeEmbedding(config, config.srcdict,
        --    config.dropout_src)({tokens, positions}):annotate({name = 'encoderEmbed'})
        local cnn
        cnn = nn.Sequential()
        cnn:add(self:makeModuleName(
            nn.Bottle(
                self:makeLinear(config.nembed, config.nhids[1],
                config.dropout_src)),
            'encoderEmbedToCNN'))
        cnn:add(nn.Contiguous())
        cnn:add(self:makeEncoderStack(config))
        
        local encoder_out = cnn(embed):annotate({name = 'encoderCNN'})

        local k = (#config.nhids - config.keeplayers_enc) % config.blocklength
        local dimacc = 1
        local outputAC_table = {}
        for i = 1, k do
            --narrow, dimension is config.nhid_accs except the first one.
            local dim = config.nhid_accs[#config.nhid_accs + 1 - i]
            local encoder_out_i = nn.Narrow(3, dimacc, dim)(encoder_out)
            --add makelinear
            encoder_out_i = nn.Bottle(self:makeLinear(dim, config.nembed))(encoder_out_i)
            --add to outputAC_table
            table.insert(outputAC_table, nn.Identity()({encoder_out_i, nn.CAddTableMulConstant(math.sqrt(0.5))({encoder_out_i, embed})}))
            dimacc = dimacc + dim 
        end
        local dim = config.numhid.encIn[#config.nhid_accs_dec + 1 - k]
        local encoder_out_i = nn.Narrow(3, dimacc, dim)(encoder_out)
        encoder_out_i = nn.Bottle(self:makeLinear(dim, config.nembed))(encoder_out_i)
        table.insert(outputAC_table, nn.Identity()({encoder_out_i, nn.CAddTableMulConstant(math.sqrt(0.5))({encoder_out_i, embed})}))
        config.numtab = #outputAC_table
        return nn.gModule({sourceIn}, outputAC_table)
    end
}

FConvDenseMoptModel.makeAttention = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='index', type='number'},
    {name='input', type='number'},
    call = function(self, config, index, input)
        local attnIn = nn.Identity()()
        local targetEm, decoderState, aencoderOut = attnIn:split(3)
        local encoderOutA, encoderOutC = aencoderOut:split(2)
        local decProj = nn.Bottle(
            self:makeLinear(input, config.nembed))(
                decoderState
        ):annotate{name = 'decProj_' .. index}
        local decoderRep = nn.CAddTableMulConstant(
            math.sqrt(0.5))({decProj, targetEm})
        -- Compute attention scores using a simple dot product between encoder
        -- output and decoder states. We can compute all attention scores at
        -- once for this model.
        local scores = nn.Bottle(nn.SoftMax())(
            nn.BeamableMM(false, true)({
                decoderRep,
                encoderOutA,
            })
        ):annotate{name = 'attentionScores_' .. index}

        -- Apply attention scores to encoder output and scale output in order to
        -- reduce variance shift. S * sqrt(1/S) is the correct factor if
        -- attention scores sum to one and are distributed uniformly for a
        -- sequence of length S.
        local attnOut = nn.Squeeze(2, 2)(
            nn.SeqMultiply()({
                nn.BeamableMM()({
                    scores,
                    encoderOutC,
                }),
                encoderOutC
            })
        )
        return nn.gModule({attnIn}, {attnOut})
    end
}

FConvDenseMoptModel.makeAttentionDense = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='index', type='number'},
    {name='input', type='number'},
    call = function(self, config, index, input)
        local attnIn = nn.Identity()()
        local targetEm, decoderState, aencoderOut = attnIn:split(3)
        --local encoderOutA, encoderOutC = aencoderOut:split(2)
        local decProj = nn.Bottle(
            self:makeLinear(input, config.nembed))(
                decoderState
        ):annotate{name = 'decProj_' .. index}
        local decoderRep = nn.CAddTableMulConstant(
            math.sqrt(0.5))({decProj, targetEm})
        -- Compute attention scores using a simple dot product between encoder
        -- output and decoder states. We can compute all attention scores at
        -- once for this model.
        local attnOut_table = {}
        for i=1, config.numtab do
            local aencoderOut_ = nn.SelectTable(i)(aencoderOut)
            local encoderOutA = nn.SelectTable(1)(aencoderOut_)
            local encoderOutC = nn.SelectTable(2)(aencoderOut_)


            local scores = nn.Bottle(nn.SoftMax())(
                nn.BeamableMM(false, true)({
                    decoderRep,
                    encoderOutA,
                })
            ):annotate{name = 'attentionScores_' .. index}

            -- Apply attention scores to encoder output and scale output in order to
            -- reduce variance shift. S * sqrt(1/S) is the correct factor if
            -- attention scores sum to one and are distributed uniformly for a
            -- sequence of length S.
            local attnOut = nn.Squeeze(2, 2)(
                nn.SeqMultiply()({
                    nn.BeamableMM()({
                        scores,
                        encoderOutC,
                    }),
                    encoderOutC
                })
            )
            table.insert(attnOut_table, attnOut)
        end
        local attnOut_final = nn.CAddTableMulConstant(math.sqrt(1.0/#attnOut_table))(attnOut_table) 
        return nn.gModule({attnIn}, {attnOut_final})
    end
}

FConvDenseMoptModel.makeTargetMappingWithSelection = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        local lut = self:makeLookupTable(config, config.dict:size(),
            config.noutembed + 1)

        -- Expected input: {lmOut, targetVocab}
        return nn.Sequential()
            :add(nn.ParallelTable()
                :add(nn.AppendBias())
                :add(lut))
            :add(nn.MM(false, true))
    end
}

FConvDenseMoptModel.makeDecoder = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        local decoderLM = self:makeDecoderLM(config)

        -- Final mapping to output vocab
        local input = nn.Identity()()
        local targetIn, encoderOut = input:split(2)
        local targetTokIn, targetPosIn, targetVocab
        if config.aligndict then
            targetTokIn, targetPosIn, targetVocab = targetIn:split(3)
        else
            targetTokIn, targetPosIn = targetIn:split(2)
        end

        -- XXX Wire this up in a sane way maybe?
        local lmOut = decoderLM({nn.Identity()({targetTokIn, targetPosIn}),
            encoderOut}):annotate({name = 'convlm'})

        -- several options, 
        -- opt 0: do not use dense
        -- use dense opt 1: concat lmout, lmin, att
        -- use dense opt 2: concat lmin, lmout+att --current
        -- use dense opt 3: concat lmin+lmout, att
        -- use dense opt 4: concat lmin+att, lmout
        config.numhid = calHidNum(config)
        local nhidout_real_final = config.numhid.decOut[#config.nlmhids]
        --[[
        if config.opt == 0 or #config.nlmhids%config.blocklength == 0 then
            nhidout_real_final = config.nlmhids[#config.nlmhids]
        elseif config.opt == 1 then
            nhidout_real_final = config.nlmhids[#config.nlmhids] + 2 * ((#config.nlmhids-1)%config.blocklength+1) * config.nhid_acc_dec
        elseif config.opt == 2 then 
            nhidout_real_final = config.nlmhids[#config.nlmhids] + ((#config.nlmhids-1)%config.blocklength+1) * config.nhid_acc_dec
        elseif config.opt == 3 then
            nhidout_real_final = 2 * config.nhid_acc_dec
        elseif config.opt == 4 then
            nhidout_real_final = 2 * config.nhid_acc_dec
        end
        --]]

        local outmodule = nn.Sequential()
            :add(nn.View(-1, nhidout_real_final ))
            :add(self:makeModuleName(
                    self:makeLinear(nhidout_real_final, config.noutembed),
                    'decoderToOutEmbed'))
            :add(self:makeDropout(config.dropout_out))
        

        --
        local outmodIn
        if targetVocab then
            local map = self:makeTargetMappingWithSelection(config)
            outmodule = nn.Sequential()
                :add(nn.ParallelTable()
                    :add(outmodule)
                    :add(nn.Identity()))
                :add(map)
            outmodIn = {lmOut, targetVocab}
        else
            outmodule:add(self:makeModuleName(
                self:makeLinear(config.noutembed, config.dict:size(), config.dropout_out),
                'outEmbedToSoftmax'))
            outmodIn = lmOut
        end

        local output = outmodule(outmodIn):annotate({name = 'outmodule'})
        return nn.gModule({input}, {output})
    end
}


FConvDenseMoptModel.makeDecoderLM = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        local input = nn.Identity()()
        local targetIn, encoderOut = input:split(2)
        local tokens, positions = targetIn:split(2)

        local targetEmbedmodel = self:makeEmbedding(config, config.dict, config.dropout_tgt)
        --need to use the same dictionary before sharing.
        --targetEmbedmodel:share(self.embed, 'weight', 'bias')

        --local targetEmbed = self:makeEmbedding(config, config.dict,
        --    config.dropout_tgt)({tokens, positions}):annotate({name = 'decoderEmbed'})
        local targetEmbed = targetEmbedmodel({tokens, positions}):annotate({name = 'decoderEmbed'})

        local attnlayers = attentionLayers(#config.nlmhids, config.attnlayers)
        assert(#config.nlmhids > 0)
        local lmOut = nn.Bottle(self:makeLinear(config.nembed, config.nlmhids[1],
            config.dropout_tgt))(targetEmbed):annotate({name = 'decoderEmbedToCNN'})

        config.numhid = calHidNum(config)
        for i = 1, #config.nlmhids do
            local nhidin = config.nlmhids[i - 1] or config.nlmhids[i]
            local nhidout = config.nlmhids[i]
            --local inpmap = nn.Identity()

            local nhidin_real = config.numhid.decIn[i]
            local nhidout_real = config.numhid.decOut[i]
            --[[
            if config.opt == 0 then
                nhidin_real = nhidin
                nhidout_real = nhidout 
            elseif config.opt == 1 then
                nhidin_real = nhidin + 2 * ((i-1)%config.blocklength) * config.nhid_acc_dec
                nhidout_real = nhidout + 2 * ((i-1)%config.blocklength+1) * config.nhid_acc_dec
            elseif config.opt == 2 then
                nhidin_real = nhidin + ((i-1)%config.blocklength) * config.nhid_acc_dec
                nhidout_real = nhidout + ((i-1)%config.blocklength+1) * config.nhid_acc_dec
            elseif config.opt == 3 or config.opt == 4 then
                if i%config.blocklength == 1 then
                    nhidin_real = nhidin
                else
                    nhidin_real = 2 * config.nhid_acc_dec
                end
                nhidout_real = 2 * config.nhid_acc_dec
            end
            --]]
            
            --
            if config.opt == 3 or config.opt == 4 then
                inpmap = self:makeModuleName(
                    nn.Bottle(
                        self:makeLinear(nhidin_real, config.nhid_accs_dec[i], config.dropout_hid)),
                    'decoderConv_inpmap_' .. i)
            end
            --]]

            
            -- Convolutional layer + GLU

            local convout 
            if config.opt == 0 or (i-config.keeplayers_dec) % config.blocklength == 0 then 
                convout = nhidout_real
            else
                convout = config.nhid_accs_dec[i]
            end

            local inpmap = nn.Identity()

            --
            if (config.opt == 0 and nhidin ~= nhidout)  then

                inpmap = self:makeModuleName(
                    nn.Bottle(
                        self:makeLinear(nhidin, nhidout, config.dropout_hid)),
                    'decoderConv_inpmap_' .. i)

            end
            if (i-config.keeplayers_dec) % config.blocklength == 0 then
                inpmap = self:makeModuleName(
                    nn.Bottle(
                        self:makeLinear(nhidin_real, nhidout, config.dropout_hid)),
                    'decoderConv_inpmap_' .. i)

            end
            --]]

            local lmIn = inpmap(lmOut)

            local lmConv = self:makeDecoderConvBlock(config.cudnnconv,
                nhidin_real, convout, config.bfactor, config.klmwidths[i],
                config.dropout_hid)(
                    self:makeDropout(config.dropout_hid)(lmOut)
                ):annotate{name = 'decoderConv_' .. i}

            
            --local lmAtt = inpmap(lmConv)
            -- Attention pass
            --if attnlayers[i] then
            local lmAtt
            if not config.denseatt then
                lmAtt = nn.Bottle(self:makeLinear(config.nembed, convout))(
                        self:makeAttention(config, i, convout)({
                            targetEmbed,
                            lmConv,
                            encoderOut,
                        }):annotate{
                            name = 'convAttnQuery_' .. i
                        }):annotate{
                            name = 'convAttnOut_' .. i
                        }
            else
                lmAtt = nn.Bottle(self:makeLinear(config.nembed, convout))(
                        self:makeAttentionDense(config, i, convout)({
                            targetEmbed,
                            lmConv,
                            encoderOut,
                        }):annotate{
                            name = 'convAttnQuery_' .. i
                        }):annotate{
                            name = 'convAttnOut_' .. i
                        }
            end
            --else
            --    print('no attention!!!')
            --end

            if config.densebn then
                --lmIn = 
                --lmConv = 
                lmAtt = nn.Bottle(nn.LayerNormalization(convout))(lmAtt)
            end
            --
            if config.opt == 0 or (i-config.keeplayers_dec) % config.blocklength == 0 then
                lmOut = nn.CAddTableMulConstant(math.sqrt(0.5))({lmIn, 
                    nn.CAddTableMulConstant(math.sqrt(0.5))({lmAtt, lmConv})})
            elseif config.opt == 1 then
                lmOut = nn.JoinTable(3)({lmIn, lmAtt, lmConv})
            elseif config.opt == 2 then
                lmOut = nn.JoinTable(3)({lmIn, 
                    nn.CAddTableMulConstant(math.sqrt(0.5))({lmAtt, lmConv})})
            elseif config.opt == 3 then
                lmOut = nn.JoinTable(3)({lmAtt,
                    nn.CAddTableMulConstant(math.sqrt(0.5))({lmIn, lmConv})})
            elseif config.opt == 4 then
                lmOut = nn.JoinTable(3)({lmConv,
                    nn.CAddTableMulConstant(math.sqrt(0.5))({lmIn, lmAtt})})
            end
            --]]
            
            -- residual connection
            --lmOut = nn.CAddTableMulConstant(math.sqrt(0.5))({lmIn, lmOut})
            --lmOut = nn.JoinTable(3)({lmOut, lmIn})
        end

        return nn.gModule({input}, {lmOut})
    end
}


FConvDenseMoptModel.make = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        assert(#config.nlmhids == #config.klmwidths)
        assert(#config.nhids == #config.kwidths)

        local encoder
        if not config.denseatt then
            encoder = self:makeEncoder(config)
        else
            encoder = self:makeEncoderDense(config)
        end
        local decoder = self:makeDecoder(config)

        -- Wire up encoder and decoder
        local input = nn.Identity()()
        local targetIn, sourceIn = input:split(2)
        local output = decoder({
            targetIn,
            encoder(sourceIn):annotate{name = 'encoder'},
        }):annotate{name = 'decoder'}

        return nn.gModule({input}, {output})
    end
}

FConvDenseMoptModel.prepareSample = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    call = function(self)
        -- Device buffers for samples
        local buffers = {
            targetIn = torch.Tensor():type(self:type()),
            targetPosIn = torch.Tensor():type(self:type()),
            target = torch.Tensor():type(self:type()),
            targetVocab = torch.Tensor():type(self:type()),
        }
        local prepareSource = self:prepareSource()

        return function(sample)
            local sourceIn = prepareSource(sample)
            local targetIn = mutils.sendtobuf(sample.input:t(),
                buffers.targetIn)
            local targetPosIn = mutils.sendtobuf(sample.inputPos:t(),
                buffers.targetPosIn)

            local target = sample.target:t()
            target = mutils.sendtobuf(target, buffers.target)
                :view(target:nElement())
            sample.target = target

            if sample.targetVocab then
                local targetVocab = mutils.sendtobuf(sample.targetVocab,
                    buffers.targetVocab)
                sample.input = {{targetIn, targetPosIn, targetVocab}, sourceIn}
            else
                sample.input = {{targetIn, targetPosIn}, sourceIn}
            end
        end
    end
}

FConvDenseMoptModel.resizeCriterionWeights = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='criterion', type='nn.Criterion'},
    {name='critweights', type='torch.CudaTensor'},
    {name='sample', type='table'},
    call = function(self, criterion, critweights, sample)
        if sample.targetVocab then
            local size = sample.targetVocab:size(1)
            -- Resize criterion weights to match target vocab size
            -- Note: we only use special weights (different from 1.0)
            -- for just a few symbols (like pad), and also we guarantee
            -- that those symbols will have the same ids from batch to batch.
            -- Thus we don't have to remap anything here.
            criterion.nll.weights = critweights:narrow(1, 1, size)
        end
    end
}

FConvDenseMoptModel.generationSetup = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local m = self:network()
        local prepareSample = self:prepareSource()
        local type = self:type()
        local targetVocabBuffer = torch.Tensor():type(self:type())

        return function(sample)
            m:evaluate()

            local state = {}
            local sourceIn = prepareSample(sample)
            state.sourceIn = sourceIn
            state.inputBuf = {
                torch.Tensor():type(type),   -- input tokens
                torch.Tensor():type(type),   -- input positions
            }

            if sample.targetVocab then
                state.remapFn = function(idx) return sample.targetVocab[idx] end
                state.targetVocab = mutils.sendtobuf(
                    sample.targetVocab, targetVocabBuffer)
            end

            if self.convStates then
                state.convStates = self.convStates
                for _, cs in ipairs(state.convStates) do
                    cs:resetState()
                end
            end
            self:setBeamSize(config.beam)

            return state
        end
    end
}

FConvDenseMoptModel.generationAttention = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local m = self:network()
        local n = config.nlmhids and #config.nlmhids or 20
        local attnscores = {}
        for i = 1, n do
            local attn = mutils.findAnnotatedNode(m, 'attentionScores_' .. i)
            if attn then
                table.insert(attnscores, attn)
            end
        end
        if #attnscores == 0 then
            return nil
        end
        local sum = torch.Tensor()

        return function(state)
            -- The decoder processes past steps in parallel during generation.
            -- Select the attention scores for the last step only.
            for i, attn in ipairs(attnscores) do
                local as = attn.output
                as = as:select(2, as:size(2))
                if i == 1 then
                    sum = sum:resize(as:size()):zero():type(as:type())
                end
                sum:add(sum, as)
            end
            return sum:div(#attnscores)
        end
    end
}

FConvDenseMoptModel.generationDecode = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        local softmax = nn.SoftMax():type(self:type())
        local m = self:network()
        local convlm = mutils.findAnnotatedNode(m, 'convlm')
        local outmodule = mutils.findAnnotatedNode(m, 'outmodule')
        local maxContext = self.decoderContext
        local pad = config.dict:getPadIndex()

        return function(state, targetIn)
            if state.remapFn then
                targetIn:apply(state.remapFn)
            end

            -- Prepare decoder input.
            local input, inputPos = table.unpack(state.inputBuf)
            if input:dim() == 0 then
                local targetInView = targetIn:view(-1, 1)
                input:resizeAs(targetInView):copy(targetInView)

                inputPos:resizeAs(input):fill(pad + 1)
            else
                if state.convStates then
                    -- The convolutional LM has an explicit state;
                    -- we need only the last generated (token, position).
                    input:copy(targetIn)
                    inputPos:add(1)
                else
                    -- The convolutional LM doesn't have an explicit state;
                    -- instead, we simply buffer the relevant past input
                    -- and compute a full forward pass on each step.
                    local nbuf = input:size(2) + 1
                    if maxContext and nbuf > maxContext then
                        input:narrow(2, 1, maxContext-1):copy(
                            input:narrow(2, 2, maxContext-1)
                        )
                        input:narrow(2, maxContext, 1):copy(targetIn)
                        inputPos:add(1)
                    else
                        local newIn = torch.cat(input, targetIn, 2)
                        input:resizeAs(newIn):copy(newIn)

                        inputPos:resizeAs(newIn)
                        for i = 1, inputPos:size(2) do
                            inputPos:narrow(2, i, 1):fill(pad + i)
                        end
                    end
                end
            end

            local cout = convlm:forward({state.inputBuf, state.encoderOut})
            cout = cout:narrow(2, cout:size(2), 1) -- take last step
            local softmaxIn = (state.softmaxIn or cout.new())
                :resizeAs(cout):copy(cout)         -- contiguous buffer
            if state.targetVocab then
                softmaxIn = {softmaxIn, state.targetVocab}
            end
            return softmax:forward(outmodule:forward(softmaxIn))
        end
    end
}

FConvDenseMoptModel.generationUpdate = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    {name='bsz', type='number'},
    call = function(self, config, bsz)
        return function(state, indexH)
            state.inputBuf[1]:copy(state.inputBuf[1]:index(1, indexH))
            -- XXX should not be necessary
            state.inputBuf[2]:copy(state.inputBuf[2]:index(1, indexH))

            if state.convStates then
                for _, cs in ipairs(state.convStates) do
                    cs:shiftState(indexH)
                end
            end
        end
    end
}

FConvDenseMoptModel.generationFinalize = argcheck{
    {name='self', type='FConvDenseMoptModel'},
    {name='config', type='table'},
    call = function(self, config)
        if config.aligndict then
            return function(state, sample, results)
                local hypos, _, _ = unpack(results)
                for _, h in ipairs(hypos) do
                    h:apply(state.remapFn)
                end
                sample.target:apply(state.remapFn)
            end
        else
            return parent.generationFinalize(self, config)
        end
    end
}

function FConvDenseMoptModel:float(...)
    self.module:replace(function(m)
        if torch.isTypeOf(m, 'cudnn.TemporalConvolution') then
            return mutils.moveTemporalConvolutionToCPU(m)
        end
        return m
    end)
    return parent.float(self, ...)
end

return FConvDenseMoptModel
