disp = require 'display'
disp.configure({port = 8001})
require 'image'

local function init_train(lr, beta1)

	optimState_d = {
		learningRate = lr,
		beta1 = beta1,
	}
	optimState_g = {
		learningRate = lr,
		beta1 = beta1,
	}

	assert(net_d and net_g, "Error : Models not defined" )
	parameters_d, gradParameters_d = net_d:getParameters()
	parameters_g, gradParameters_g = net_g:getParameters()

end

function validate(valid_data, batch_size, valid_plot_x, valid_plot_y)

	local tesize = get_size(valid_data)
	local depth = opt.depth
	local im_size = opt.im_size
	local channels = opt.input_nc
	local time_step = opt.time_step
	local l1_error = 0
	
	print(color.green'\n====> Running validation')
	for t = 1, tesize, batch_size do
		xlua.progress(t, tesize)
		local num_samples = math.min(batch_size, tesize - t + 1)
	    local input_g = torch.Tensor(time_step - 1, num_samples, channels, depth, im_size, im_size):zero()
	    local real_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
	    local fake_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
    	
    	input_g:copy(valid_data[{{1, time_step-1}, {t, t+num_samples-1}}]:float())
		real_output_g:copy(valid_data[{{time_step}, {t, t+num_samples-1}}]:float())
		-- Scale 0 to 1
		if input_g:max() > 1 then
			input_g = scale_zero_one(input_g)
			real_output_g = scale_zero_one(real_output_g)
		end
		-- Scale -1 to 1 (tanh)
		input_g = input_g:mul(2):add(-1)
		real_output_g = real_output_g:mul(2):add(-1)

		if opt.gpu > 0 then 
			input_g = input_g:cuda()
			real_output_g = real_output_g:cuda()
			fake_output_g = fake_output_g:cuda()
		end

		fake_output_g = net_g:forward(input_g)
		local err_l1 = criterionAE:forward(fake_output_g, real_output_g)
		l1_error = l1_error + num_samples * err_l1
	end

	local avg_l1 = l1_error / tesize
	print(color.blue "\nValidation average L1 error : " .. avg_l1)
	if opt.plot == 1 then
		table.insert(valid_plot_y, avg_l1 * 100)
		gnuplot.figure(2)
		gnuplot.title('Validation L1 Loss')
		gnuplot.plot('L1 Error', torch.Tensor(valid_plot_x), torch.Tensor(valid_plot_y),'-')
		gnuplot.plotflush()
	end

	return valid_plot_y
end

function train(train_image, train_data, num_epochs, batch_size, valid_data)

	local real_label = 1
	local fake_label = 0	
	local depth = opt.depth
	local im_size = opt.im_size
	local channels = opt.input_nc
	local time_step = opt.time_step
	local err_d, err_g, err_l1 = 0, 0, 0
	local trsize = get_size(train_data)
	local epoch_tm = torch.Timer()
	local folder_path = paths.concat('/home/koustav.m/models/fpgan/', opt.save_folder)
	paths.mkdir(folder_path)
	local file = torch.DiskFile(paths.concat('/home/koustav.m/models/fpgan/', opt.save_folder, 'stats.txt'), 'w')
	if opt.plot == 1 then
		train_plot_x = {}
		train_plot_y = {}
		if opt.validate == 1 then
			valid_plot_x = {}
			valid_plot_y = {}
		end
	end
	local max_l1 = math.huge
	
	if opt.gpu > 0 then 
		
		if opt.cudnn == 1 then
      		net_d = cudnn_convert_custom(net_d, cudnn);
      		net_g = cudnn_convert_custom(net_g, cudnn); 
		end		

		net_d:cuda();
		net_g:cuda();
		criterion:cuda();
		criterionAE:cuda();
	end
	
	init_train(opt.lr, opt.beta1)
	local idx = 0

	for epoch = 1, num_epochs do

		-- Unsure whether to use or not
		-- net_d:training()
		-- net_g:training()
		
		local counter = 0
		local l1_error = 0
		local shuffle = torch.randperm(trsize)
		epoch_tm:reset()

		-- idx = idx + 1
		-- local save_image = torch.Tensor(1, 128, 128):zero()
		-- local input = train_data[{{}, {1050}}]:clone():float()
		-- input = scale_zero_one(input)
		-- input = input:mul(2):add(-1)
		-- local output = net_g:forward(input[{{1, time_step-1}}]:cuda())
		-- output = deprocess(output:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float()) / 255
		-- save_image[{{}, {1, 64}, {1, 64}}] = output[1]
		-- save_image[{{}, {1, 64}, {65, 128}}] = output[2]
		-- save_image[{{}, {65, 128}, {1, 64}}] = output[3]
		-- save_image[{{}, {65, 128}, {65, 128}}] = output[4]
		-- image.save('train_images/output_' .. idx .. '.png', image.scale(save_image, 256, 256))

		-- input = train_image:float()
		-- input = scale_zero_one(input)
		-- input = input:mul(2):add(-1)
		-- output = net_g:forward(input[{{1, time_step-1}}]:cuda())
		-- output = deprocess(output:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float()) / 255
		-- save_image[{{}, {1, 64}, {1, 64}}] = output[1]
		-- save_image[{{}, {1, 64}, {65, 128}}] = output[2]
		-- save_image[{{}, {65, 128}, {1, 64}}] = output[3]
		-- save_image[{{}, {65, 128}, {65, 128}}] = output[4]
		-- image.save('test_images/output_' .. idx .. '.png', image.scale(save_image, 256, 256))

	    for t = 1, trsize, batch_size do
			
    		xlua.progress(t % opt.print_freq, opt.print_freq)
    		local num_samples = math.min(batch_size, trsize - t + 1)
		    local real_input_d = torch.Tensor(time_step, num_samples, channels, depth, im_size, im_size):zero()
		    local fake_input_d = torch.Tensor(time_step, num_samples, channels, depth, im_size, im_size):zero()
		    local input_g = torch.Tensor(time_step - 1, num_samples, channels, depth, im_size, im_size):zero()
		    local real_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
		    local fake_output_g = torch.Tensor(num_samples, channels, depth, im_size, im_size):zero()
		    local k = 1
		    
		    for i = t, math.min(t+batch_size-1, trsize) do
	      		real_input_d[{{}, {k}}]:copy(train_data[{{}, {shuffle[i]}}]:float())
	      		k = k+1
		    end

			-- Scale 0 to 1
			if real_input_d:max() > 1 then
				real_input_d = scale_zero_one(real_input_d)
			end
			-- Scale -1 to 1 (tanh)
			real_input_d = real_input_d:mul(2):add(-1)
			input_g = real_input_d[{{1, time_step - 1}}]
			real_output_g = real_input_d[time_step]:reshape(real_output_g:size())
			fake_input_d[{{1, time_step - 1}}] = real_input_d[{{1, time_step - 1}}]
			
			if opt.gpu > 0 then 
				real_input_d = real_input_d:cuda()
				fake_input_d = fake_input_d:cuda()
				input_g = input_g:cuda()
				real_output_g = real_output_g:cuda()
				fake_output_g = fake_output_g:cuda()
			end
			
			fake_output_g = net_g:forward(input_g)
			fake_input_d[time_step] = fake_output_g

			local last_input = fake_output_g[1]:clone()
		    last_input = deprocess(last_input:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float())
		    disp.image(last_input, {win=12, title='Fake'})
		    last_input = real_output_g[1]:clone()
		    last_input = deprocess(last_input:squeeze():reshape(opt.depth, opt.input_nc, opt.im_size, opt.im_size):float())
		    disp.image(last_input, {win=13, title='Real'})
		    
			local fDx = function(x)
			    
			    net_d:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
			    net_g:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
			    
			    gradParameters_d:zero()
			    
			    -- Noisy labels
			    if opt.label_flip == 1 then
				    if torch.uniform() < opt.flip_prob then
				    	real_label = 0
				    	fake_label = 1
				    end
				end
			    -- Real
			    local output = net_d:forward(real_input_d)
			    local label = torch.FloatTensor(output:size()):fill(real_label)
			    if opt.gpu > 0 then 
			    	label = label:cuda()
			    end
			    local err_d_real = criterion:forward(output, label)
			    local df_do = criterion:backward(output, label)
			    net_d:backward(real_input_d, df_do)
			    
			    -- Fake
			    local output = net_d:forward(fake_input_d)
			    label:fill(fake_label)
			    local err_d_fake = criterion:forward(output, label)
			    local df_do = criterion:backward(output, label)
			    net_d:backward(fake_input_d, df_do)
			    err_d = (err_d_real + err_d_fake)/2
		    
		    	-- Reset labels if flipped
		    	if opt.label_flip == 1 then
					real_label = 1
		    		fake_label = 0
		    	end
			    
			    return err_d, gradParameters_d
			end

			local fGx = function(x)
			    
			    net_d:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
			    net_g:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
			    
			    gradParameters_g:zero()
			    
			    -- GAN loss
			    
		       	local df_dg = torch.zeros(fake_output_g:size())
			    if opt.gpu > 0 then 
			    	df_dg = df_dg:cuda();
			    end
			    
			    -- Backprop resets the step size of D
		       	-- Need to re-run again. Cannot use net_d.output from D
				local output = net_d:forward(fake_input_d)
		       	local label = torch.FloatTensor(output:size()):fill(real_label)
		       	
		       	if opt.gpu > 0 then 
		       		label = label:cuda();
		       	end
		       	err_g = criterion:forward(output, label)
		       	local df_do = criterion:backward(output, label)
		       	df_dg = net_d:updateGradInput(fake_input_d, df_do):select(1, 1)
		       	
		       	-- unary loss
			    local df_do_AE = torch.zeros(fake_output_g:size())
			    if opt.gpu > 0 then 
			    	df_do_AE = df_do_AE:cuda();
			    end
		     	err_l1 = criterionAE:forward(fake_output_g, real_output_g)
		     	df_do_AE = criterionAE:backward(fake_output_g, real_output_g)
			    
			    net_g:backward(input_g, df_dg + df_do_AE:mul(opt.lambda))
			    
			    return err_g, gradParameters_g
			end

			optim.adam(fDx, parameters_d, optimState_d)
        	optim.adam(fGx, parameters_g, optimState_g)
        	l1_error = l1_error + num_samples * err_l1
        	counter = counter + num_samples
        	collectgarbage()

        	if counter >= opt.print_freq then
        		local file = torch.DiskFile(paths.concat('/home/koustav.m/models/fpgan/', opt.save_folder, 'stats.txt'), 'rw')
        		local op = "Epoch : ".. epoch.." Generator : "..err_g.." Discriminator : "..err_d.." L1: "..err_l1.."\n"
				file:seekEnd()
				file:writeString(op)
				file:close()
				print(color.magenta "\nEpoch : " .. epoch .. color.magenta " Generator error : " .. err_g 
        			.. color.magenta " Discriminator error : " .. err_d .. color.magenta " L1 error : " .. err_l1)
        		counter = counter % opt.print_freq
			end
        end

		parameters_d, gradParameters_d = nil, nil
		parameters_g, gradParameters_g = nil, nil
    	parameters_d, gradParameters_d = net_d:getParameters()
		parameters_g, gradParameters_g = net_g:getParameters()

		local avg_l1 = l1_error / trsize
		print(color.blue "\nAverage L1 error : " .. avg_l1)
		if opt.plot == 1 then
			table.insert(train_plot_x, epoch)
			table.insert(train_plot_y, avg_l1 * 100)
			gnuplot.figure(1)
			gnuplot.title('Training L1 Loss')
			gnuplot.plot('L1 Error', torch.Tensor(train_plot_x), torch.Tensor(train_plot_y),'-')
			gnuplot.plotflush()
		end
		if avg_l1 < max_l1 then
			max_l1 = avg_l1
			if opt.save == 1 then
				torch.save(folder_path .. '/discriminator.t7', net_d:clearState())
				torch.save(folder_path .. '/generator.t7', net_g:clearState())
				print(color.red "Models saved")
			end
		end
		
		if opt.validate == 1 then
			table.insert(valid_plot_x, epoch)
			valid_plot_y = validate(valid_data, batch_size, valid_plot_x, valid_plot_y)
		end
		print(color.cyan('\nEnd of epoch %d / %d \t Time Taken: %.3f\n'):format(epoch, num_epochs, epoch_tm:time().real))

	end

	-- torch.save(folder_path .. '/discriminator.t7', net_d:clearState())
	-- torch.save(folder_path .. '/generator.t7', net_g:clearState())
	-- print(color.red "\nModels saved")
end