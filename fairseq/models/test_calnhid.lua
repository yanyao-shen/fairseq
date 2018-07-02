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

config = {}
config['opt'] = 1
config['optenc'] = 1
config['nhids'] = {256,256,256,256,256,256,256,256,256,256,256,256.256}
config['nlmhids'] = {256,256,256,256,256,256,256,256,256,256,256,256,256}
config['keeplayers_enc'] = 0
config['keeplayers_dec'] = 0
config['blocklength'] = 5
config['nhid_accs'] = {16,16,16,16,16,16,16,16,16,16,16,16,16}
config['nhid_accs_dec'] = {16,16,16,16,16,16,16,16,16,16,16,16,16}

res = calHidNum(config)
print(res)

