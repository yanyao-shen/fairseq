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

local FConvDenseModel, parent = torch.class('FConvDenseModel', 'AvgpoolModel')

FConvDenseModel.__init = argcheck{
    {name='self', type='Model'},
    {name='config', type='table', opt=true},
    call = function(self, config)
        parent.__init(self, config)

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

local ConvDenseBlockTrainTestLayer, _ = torch.class('nn.ConvDenseBlockTrainTestLayer',
    'nn.TrainTestLayer')

function ConvDenseBlockTrainTestLayer:__init(trainModule, evalModule)
    nn.Container.__init(self)
    self.modules[1] = trainModule
    self.modules[2] = evalModule
    self.train = true
end

ConvDenseBlockTrainTestLayer.onTrain = function() end
ConvDenseBlockTrainTestLayer.onEvaluate = function(trnMod, testMod)
    local conv = trnMod:get(2):get(1)
    testMod:setParameters(
        conv.weight:transpose(3, 1):transpose(2, 3),
        conv.bias
    )
end

FConvDenseModel.makeDecoderFast = argcheck{
    {name='self', type='Model'},
    call = function(self)
        if self.convStates then return end

        -- edit the conv blocks to use LinearizedConvolution at test time
        local decoder = mutils.findAnnotatedNode(self:network(), 'decoder')
        for i = 1, math.huge do
            local block = mutils.findAnnotatedNode(decoder, 'ConvDenseBlock_' .. i)
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
            local trnTest = nn.ConvDenseBlockTrainTestLayer(trnMod, testMod)
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

FConvDenseModel.setBeamSize = argcheck{
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

FConvDenseModel.makeLookupTable = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeEmbedding = argcheck{
    {name='self', type='FConvDenseModel'},
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
        embed:add(nn.ParallelTable()
            :add(nn.Sequential()
                :add(embedTokens)
                :add(nn.Bottle(self:makeLinear(config.ninembed, config.nembed, config.dropout_src))) )
            :add(nn.Sequential()
                :add(embedPositions)
                :add(nn.Bottle(self:makeLinear(config.ninembed, config.nembed, config.dropout_src))) )
        )
        embed:add(nn.CAddTable())
        embed:add(self:makeDropout(dropout))
        return embed
    end
}

FConvDenseModel.makeTemporalConv = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeBottleneckConv = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeTranspose = argcheck{
    {name='self', type='FConvDenseModel'},
    {name='cudnnconv', type='boolean'},
    call = function(self, cudnnconv)
        if cudnnconv then
            return nn.Identity()
        else
            return nn.Transpose({1, 2})
        end
    end
}

FConvDenseModel.makeEncoderConvBlock = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeDecoderConvBlock = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeLinear = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeDropout = argcheck{
    {name='self', type='FConvDenseModel'},
    {name='p', type='number', opt=true},
    call = function(self, p)
        if not p or p == 0 then
            return nn.Identity()
        end
        return nn.Dropout(p)
    end
}

FConvDenseModel.makeModuleName = argcheck{
    {name='self', type='FConvDenseModel'},
    {name='module', type='nn.Module'},
    {name='name', type='string'},
    call = function(self, module, name)
        module.name = name
        return module
    end
}

FConvDenseModel.makeEncoderStack = argcheck{
    {name='self', type='FConvDenseModel'},
    {name='config', type='table'},
    {name='nhids', type='table'},
    {name='nhid_acc', type='number'},
    call = function(self, config, nhids, nhid_acc)
        if #nhids == 0 then
            return nn.Identity()
        end

        local stack = nn.Sequential()
        stack:add(self:makeTranspose(config.cudnnconv))
        --local nhid_acc = 256
        for i = 1, #nhids do
            if i % 5 ~= 0 then
                local nhidin = nhids[i - 1] or nhids[i]
                local nhidout = nhids[i]

                local nhidin_real = nhidin + ((i-1)%5) * nhid_acc
                local nhidout_real = nhidout + ((i-1)%5+1) * nhid_acc


                local inpmap = nn.Identity()
                --[[
                if nhidin ~= nhidout then
                    inpmap = self:makeModuleName(
                        nn.Bottle(
                            self:makeLinear(nhidin, nhidout, config.dropout_hid)),
                        'encoderConv_inpmap_' .. i)
                end
                --]]

                -- Residual connections
                stack:add(nn.Concat(3)
                    :add(nn.Sequential()
                        :add(self:makeDropout(config.dropout_hid))
                        :add(self:makeModuleName(
                                self:makeEncoderConvBlock(config.cudnnconv,
                                nhidin_real, nhid_acc, config.bfactor, config.kwidths[i],
                                config.dropout_hid),
                                'encoderConv_' .. i)))
                    :add(inpmap))
            else
                local nhidin = nhids[i-1] or nhids[i]
                local nhidout = nhids[i]

                local nhidin_real = nhidin + ((i-1)%5) * nhid_acc
                local nhidout_real = nhidout
                stack:add(nn.Sequential()
                    :add(self:makeDropout(config.dropout_hid))
                    :add(self:makeModuleName(
                            self:makeEncoderConvBlock(config.cudnnconv,
                            nhidin_real, nhidout_real, config.bfactor, config.kwidths[i],
                            config.dropout_hid),
                            'encoderConv_' .. i)))
            end
        end
        stack:add(self:makeTranspose(config.cudnnconv))
        return stack
    end
}

FConvDenseModel.makeEncoder = argcheck{
    {name='self', type='FConvDenseModel'},
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
        if #config.nhids > 0 then
            cnn = nn.Sequential()
            cnn:add(self:makeModuleName(
                nn.Bottle(
                    self:makeLinear(config.nembed, config.nhids[1],
                    config.dropout_src)),
                'encoderEmbedToCNN'))
            cnn:add(nn.Contiguous())
            cnn:add(self:makeEncoderStack(config, config.nhids, config.nhid_acc))
            
            local nhidout_real = ((#config.nhids-1)%5+1) * config.nhid_acc + config.nhids[#config.nhids]
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

FConvDenseModel.makeAttention = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeTargetMappingWithSelection = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.makeDecoder = argcheck{
    {name='self', type='FConvDenseModel'},
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
        
        local nhidout_real_final = config.nlmhids[#config.nlmhids] + ((#config.nlmhids-1)%5+1) * config.nhid_acc_dec

        local outmodule = nn.Sequential()
            :add(nn.View(-1, nhidout_real_final ))
            :add(self:makeModuleName(
                    self:makeLinear(nhidout_real_final, config.noutembed),
                    'decoderToOutEmbed'))
            :add(self:makeDropout(config.dropout_out))
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


FConvDenseModel.makeDecoderLM = argcheck{
    {name='self', type='FConvDenseModel'},
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

        
        for i = 1, #config.nlmhids do
            if i%5 ~= 0 then
                local nhidin = config.nlmhids[i - 1] or config.nlmhids[i]
                local nhidout = config.nlmhids[i]
                local inpmap = nn.Identity()



                local nhidin_real = nhidin + ((i-1)%5) * config.nhid_acc_dec
                local nhidout_real = nhidout + ((i-1)%5+1) * config.nhid_acc_dec
                
                --[[
                if nhidin ~= nhidout then
                    inpmap = self:makeModuleName(
                        nn.Bottle(
                            self:makeLinear(nhidin, nhidout, config.dropout_hid)),
                        'decoderConv_inpmap_' .. i)
                end
                --]]

                local lmIn = inpmap(lmOut)
                -- Convolutional layer + GLU
                lmOut = self:makeDecoderConvBlock(config.cudnnconv,
                    nhidin_real, config.nhid_acc_dec, config.bfactor, config.klmwidths[i],
                    config.dropout_hid)(
                        self:makeDropout(config.dropout_hid)(lmOut)
                    ):annotate{name = 'decoderConv_' .. i}

                -- Attention pass
                if attnlayers[i] then
                    lmOut = nn.CAddTableMulConstant(math.sqrt(0.5))({
                        nn.Bottle(self:makeLinear(config.nembed, config.nhid_acc_dec))(
                            self:makeAttention(config, i, config.nhid_acc_dec)({
                                targetEmbed,
                                lmOut,
                                encoderOut,
                            }):annotate{
                                name = 'convAttnQuery_' .. i
                            }):annotate{
                                name = 'convAttnOut_' .. i
                            },
                        lmOut,
                    })
                else
                    print('no attention!!!')
                end

                -- residual connection
                --lmOut = nn.CAddTableMulConstant(math.sqrt(0.5))({lmIn, lmOut})
                lmOut = nn.JoinTable(3)({lmOut, lmIn})
            else

                local nhidin = config.nlmhids[i - 1] or config.nlmhids[i]
                local nhidout = config.nlmhids[i]
                local inpmap = nn.Identity()



                local nhidin_real = nhidin + ((i-1)%5) * config.nhid_acc_dec
                local nhidout_real = nhidout

                if nhidin_real ~= nhidout_real then
                    inpmap = self:makeModuleName(
                        nn.Bottle(
                            self:makeLinear(nhidin_real, nhidout_real, config.dropout_hid)),
                        'decoderConv_inpmap_' .. i)
                end
                local lmIn = inpmap(lmOut)

                -- Convolutional layer + GLU
                lmOut = self:makeDecoderConvBlock(config.cudnnconv,
                    nhidin_real, nhidout_real, config.bfactor, config.klmwidths[i],
                    config.dropout_hid)(
                        self:makeDropout(config.dropout_hid)(lmOut)
                    ):annotate{name = 'decoderConv_' .. i}

                -- Attention pass
                if attnlayers[i] then
                    lmOut = nn.CAddTableMulConstant(math.sqrt(0.5))({
                        nn.Bottle(self:makeLinear(config.nembed, nhidout_real))(
                            self:makeAttention(config, i, nhidout_real)({
                                targetEmbed,
                                lmOut,
                                encoderOut,
                            }):annotate{
                                name = 'convAttnQuery_' .. i
                            }):annotate{
                                name = 'convAttnOut_' .. i
                            },
                        lmOut,
                    })
                else
                    print('no attention!!!')
                end

                -- residual connection
                lmOut = nn.CAddTableMulConstant(math.sqrt(0.5))({lmIn, lmOut})
                --lmOut = nn.JoinTable(3)({lmOut, lmIn})
            end
        end

        return nn.gModule({input}, {lmOut})
    end
}

FConvDenseModel.make = argcheck{
    {name='self', type='FConvDenseModel'},
    {name='config', type='table'},
    call = function(self, config)
        assert(#config.nlmhids == #config.klmwidths)
        assert(#config.nhids == #config.kwidths)

        local encoder = self:makeEncoder(config)
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

FConvDenseModel.prepareSample = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.resizeCriterionWeights = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.generationSetup = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.generationAttention = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.generationDecode = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.generationUpdate = argcheck{
    {name='self', type='FConvDenseModel'},
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

FConvDenseModel.generationFinalize = argcheck{
    {name='self', type='FConvDenseModel'},
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

function FConvDenseModel:float(...)
    self.module:replace(function(m)
        if torch.isTypeOf(m, 'cudnn.TemporalConvolution') then
            return mutils.moveTemporalConvolutionToCPU(m)
        end
        return m
    end)
    return parent.float(self, ...)
end

return FConvDenseModel
