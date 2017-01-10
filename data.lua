mattorch = require 'mattorch'

function get_size(data)
	return data:size(2)
end

function scale_zero_one(data)
	return (data / 255)
end

function deprocess(data)
	return data:add(1):div(2):mul(255)
end

function convert_tsteps(data)
	data = data:reshape(opt.time_step, opt.depth, get_size(data), opt.input_nc, opt.im_size, opt.im_size)
	return data:permute(1, 3, 4, 2, 5, 6)
end

function get_data(data_path)

	local data = mattorch.load(data_path)

	-- Data dimensions are 20 x n x 64 x 64
	-- Need to convert to 5 x n x 1 x 4 x 64 x 64 (time_step x n x input_nc x depth x im_size x im_size)
	local train_data = convert_tsteps(data.train_data:permute(4, 3, 2, 1))
	local valid_data = convert_tsteps(data.valid_data:permute(4, 3, 2, 1))
	local test_data = convert_tsteps(data.test_data:permute(4, 3, 2, 1))
	
	return train_data, valid_data, test_data
end
	