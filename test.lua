require 'nn'
require 'rnn'
require 'dpnn'
disp = require 'display'
disp.configure({port = 8001})
color = require 'trepl.colorize'

opt = {
	manualSeed = 1,
	input_nc = 1,
	output_nc = 1,
	im_size = 64,
	depth = 4,
	time_step = 5,
	batch_size = 64,
	data_path = '/home/koustav.m/data/moving_mnist/moving_mnist.mat',
	gpu = 1,
	cudnn = 1,
	save_folder = 'normal',
	display = 1,
	display_id = 20,
	disp_model = 0
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpu > 0 then
	require 'cunn'
end
if opt.cudnn == 1 then
	require 'cudnn'
end

local function deprocess(data)
	return data:add(1):div(2):mul(255)
end

dofile 'data.lua'
print(color.green'\n====> Loading data')
local train_data, valid_data, test_data = get_data(opt.data_path)
print(color.green'\n====> Data load complete')
local data = test_data

local folder_path = paths.concat('/home/koustav.m/models/fpgan/', opt.save_folder)
print(color.green'\n====> Loading Generator')
local g_model = torch.load(folder_path .. '/generator.t7')
print(color.green'\n====> Generator load complete')
if opt.disp_model == 1 then 
	print(g_model)
end

local idx = torch.random(1, get_size(data))
-- local idx = 50
local input = data[{{1, opt.time_step - 1}, {idx}}]:float()
local gt = data[{{opt.time_step}, {idx}}]:float()

if input:max() > 1 then
	input = scale_zero_one(input:float())
end
-- Scale -1 to 1 (tanh)
input = input:mul(2):add(-1)

if opt.gpu > 0 then 
	input = input:cuda()
	g_model:cuda()
end
print(color.green'\n====> Generating output')
local output = g_model:forward(input)
local last_input = deprocess(input[opt.time_step - 1]:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float())
gt = gt:squeeze():reshape(opt.depth, opt.output_nc, opt.im_size, opt.im_size):float() -- Not scaled -1 to 1
output = deprocess(output:squeeze():reshape(opt.depth, opt.output_nc, opt.im_size, opt.im_size):float())

if opt.display == 1 then
	disp.image(last_input, {win = opt.display_id, title = 'Input'})
	disp.image(gt, {win = opt.display_id + 1, title = 'GT'})
	disp.image(output, {win = opt.display_id + 2, title = 'Output'})
end
