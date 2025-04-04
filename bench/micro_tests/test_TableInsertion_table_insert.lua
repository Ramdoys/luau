local function prequire(name) local success, result = pcall(require, name); return success and result end
local bench = script and require(script.Parent.bench_support) or prequire("bench_support") or require("../bench_support")

function test()

    local ts0 = os.clock()

    for i=1,300 do
        local t = {}
        for j=1,1000 do
            table.insert(t, j)
        end
    end

    local ts1 = os.clock()

    return ts1-ts0
end

bench.runCode(test, "TableInsertion: table.insert")