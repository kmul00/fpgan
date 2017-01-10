require 'nn'
require 'rnn'
require 'dpnn'
require 'optim'
require 'xlua'
require 'gnuplot'

opt = {
	manualSeed = 1,
	input_nc = 1,
	output_nc = 1,
	im_size = 64,
	depth = 4,
	time_step = 5,
	ngf = 64,
	ndf = 96,
	lr = 0.0005,
	beta1 = 0.5,
	lambda = 1000,
	num_epochs = 100,
	batch_size = 64,
	data_path = '/home/koustav.m/data/moving_mnist/moving_mnist.mat',
	disp_model = 0,
	print_freq = 3500,
	neg_val = 0.2,
	input_noise = 0,
	noise_mean = 0,
	noise_std = 0.4,
	label_flip = 0,
	flip_prob = 0.4,
	gpu = 1,
	cudnn = 1,
	save_folder = 'temp',
	save = 1,
	plot = 1,
	validate = 1
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

if opt.gpu > 0 then
	require 'cunn'
end
if opt.cudnn == 1 then
	require 'cudnn'
end
