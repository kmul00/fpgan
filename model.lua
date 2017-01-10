function weights_init(m)
   local name = torch.type(m)

   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)

   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end

end

function get_generator_model(input_nc, output_nc, ngf)

	local neg_val = opt.neg_val
	local encoder = nn.Sequential()
	local lstm = nn.Sequential()
	local decoder = nn.Sequential()
	local g_model = nn.Sequential()
	
	-- input is (nc) x 4 x 64 x 64
	encoder:add(nn.VolumetricConvolution(input_nc, ngf, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(nn.LeakyReLU(neg_val, true))

	-- input is (ngf) x 4 x 32 x 32
	encoder:add(nn.VolumetricConvolution(ngf, ngf*2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ngf*2)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*2) x 2 x 16 x 16
	encoder:add(nn.VolumetricConvolution(ngf*2, ngf*4, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ngf*4)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*4) x 2 x 8 x 8
	encoder:add(nn.VolumetricConvolution(ngf*4, ngf*8, 4, 4, 4, 2, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ngf*8)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*8) x 1 x 4 x 4
	encoder:add(nn.VolumetricConvolution(ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ngf*8)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*8) x 1 x 2 x 2
	encoder:add(nn.VolumetricConvolution(ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ngf*8)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*8) x 1 x 1 x 1
	encoder:add(nn.View(-1):setNumInputDims(4))
	
	-- input is (ngf*8)
	lstm:add(nn.LSTM(ngf*8, ngf*4))
	
	-- input is (ngf*4)
	decoder:add(nn.Reshape(ngf*4, 1, 1, 1)):add(nn.ReLU(true))

	-- input is (ngf*4) x 1 x 1 x 1
	decoder:add(nn.VolumetricConvolution(ngf*4, ngf*8, 1, 1, 1))
	decoder:add(nn.VolumetricBatchNormalization(ngf*8)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*8) x 1 x 1 x 1
	decoder:add(nn.VolumetricFullConvolution(ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	decoder:add(nn.VolumetricBatchNormalization(ngf*8)):add(nn.Dropout(0.5)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*8) x 1 x 2 x 2
	decoder:add(nn.VolumetricFullConvolution(ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	decoder:add(nn.VolumetricBatchNormalization(ngf*8)):add(nn.Dropout(0.5)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*8) x 1 x 4 x 4
	decoder:add(nn.VolumetricFullConvolution(ngf*8, ngf*4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
	decoder:add(nn.VolumetricBatchNormalization(ngf*4)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*4) x 2 x 8 x 8
	decoder:add(nn.VolumetricFullConvolution(ngf*4, ngf*2, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	decoder:add(nn.VolumetricBatchNormalization(ngf*2)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf*2) x 2 x 16 x 16
	decoder:add(nn.VolumetricFullConvolution(ngf*2, ngf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
	decoder:add(nn.VolumetricBatchNormalization(ngf)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ngf) x 4 x 32 x 32
	decoder:add(nn.VolumetricFullConvolution(ngf, output_nc, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	decoder:add(nn.Tanh())
	
	-- output is (nc) x 4 x 64 x 64

	g_model:add(nn.Sequencer(encoder))
	g_model:add(nn.Sequencer(lstm))
	g_model:add(nn.Select(1, -1))
	g_model:add(decoder)

	return g_model
end

function get_generator_u_model(input_nc, output_nc, ngf)

	local neg_val = opt.neg_val
	local lstm = nn.LSTM(ngf*8, ngf*4)

	local middle = nn.Sequential()
	middle:add(nn.Sequencer(nn.Sequential():add(nn.LeakyReLU(neg_val, true)):add(nn.View(-1):setNumInputDims(4))))
	middle:add(nn.Sequencer(lstm))
	middle:add(nn.Select(1, -1))
	middle:add(nn.Reshape(ngf*4, 1, 1, 1)):add(nn.LeakyReLU(neg_val, true))
	middle:add(nn.VolumetricConvolution(ngf*4, ngf*8, 1, 1, 1))
	middle:add(nn.VolumetricBatchNormalization(ngf*8))

	local enc_6 = nn.Sequential()
	enc_6:add(nn.Sequencer(nn.Sequential():add(nn.LeakyReLU(neg_val, true)):add(
		nn.VolumetricConvolution(ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*8))))
	enc_6:add(nn.ConcatTable():add(middle):add(nn.Select(1, -1)))

	local dec_1 = nn.Sequential():add(nn.JoinTable(2))
	dec_1:add(nn.LeakyReLU(neg_val, true)):add(nn.VolumetricFullConvolution(2*ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*8)):add(nn.Dropout(0.5))

	local block_6 = nn.Sequential():add(enc_6):add(dec_1)

	local enc_5 = nn.Sequential()
	enc_5:add(nn.Sequencer(nn.Sequential():add(nn.LeakyReLU(neg_val, true)):add(
		nn.VolumetricConvolution(ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*8))))
	enc_5:add(nn.ConcatTable():add(block_6):add(nn.Select(1, -1)))

	local dec_2 = nn.Sequential():add(nn.JoinTable(2))
	dec_2:add(nn.LeakyReLU(neg_val, true)):add(nn.VolumetricFullConvolution(2*ngf*8, ngf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*8)):add(nn.Dropout(0.5))

	local block_5 = nn.Sequential():add(enc_5):add(dec_2)
	
	local enc_4 = nn.Sequential()
	enc_4:add(nn.Sequencer(nn.Sequential():add(nn.LeakyReLU(neg_val, true)):add(
		nn.VolumetricConvolution(ngf*4, ngf*8, 4, 4, 4, 2, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*8))))
	enc_4:add(nn.ConcatTable():add(block_5):add(nn.Select(1, -1)))

	local dec_3 = nn.Sequential():add(nn.JoinTable(2))
	dec_3:add(nn.LeakyReLU(neg_val, true)):add(nn.VolumetricFullConvolution(2*ngf*8, ngf*4, 4, 4, 4, 2, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*4))
	
	local block_4 = nn.Sequential():add(enc_4):add(dec_3)

	local enc_3 = nn.Sequential()
	enc_3:add(nn.Sequencer(nn.Sequential():add(nn.LeakyReLU(neg_val, true)):add(
		nn.VolumetricConvolution(ngf*2, ngf*4, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*4))))
	enc_3:add(nn.ConcatTable():add(block_4):add(nn.Select(1, -1)))

	local dec_4 = nn.Sequential():add(nn.JoinTable(2))
	dec_4:add(nn.LeakyReLU(neg_val, true)):add(nn.VolumetricFullConvolution(2*ngf*4, ngf*2, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*2))

	local block_3 = nn.Sequential():add(enc_3):add(dec_4)

	local enc_2 = nn.Sequential()
	enc_2:add(nn.Sequencer(nn.Sequential():add(nn.LeakyReLU(neg_val, true)):add(
		nn.VolumetricConvolution(ngf, ngf*2, 4, 4, 4, 2, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf*2))))
	enc_2:add(nn.ConcatTable():add(block_3):add(nn.Select(1, -1)))

	local dec_5 = nn.Sequential():add(nn.JoinTable(2))
	dec_5:add(nn.LeakyReLU(neg_val, true)):add(nn.VolumetricFullConvolution(2*ngf*2, ngf, 4, 4, 4, 2, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf))

	local block_2 = nn.Sequential():add(enc_2):add(dec_5)

	local enc_1 = nn.Sequential()
	enc_1:add(nn.Sequencer(nn.VolumetricConvolution(input_nc, ngf, 3, 4, 4, 1, 2, 2, 1, 1, 1)))
	enc_1:add(nn.ConcatTable():add(block_2):add(nn.Select(1, -1)))

	local dec_6 = nn.Sequential():add(nn.JoinTable(2))
	dec_6:add(nn.LeakyReLU(neg_val, true)):add(nn.VolumetricFullConvolution(2*ngf, ngf, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(
		nn.VolumetricBatchNormalization(ngf))

	local final_layer = nn.Sequential()
	final_layer:add(nn.LeakyReLU(neg_val, true)):add(nn.VolumetricFullConvolution(ngf, output_nc, 3, 3, 3, 1, 1, 1, 1, 1, 1))

	local g_model = nn.Sequential():add(enc_1):add(dec_6):add(final_layer):add(nn.Tanh())

	return g_model
end

function get_discriminator_model(input_nc, output_nc, ndf)

	local neg_val = opt.neg_val
	local encoder = nn.Sequential()
	local lstm = nn.Sequential()
	local d_model = nn.Sequential()

	-- Noisy input
	if opt.input_noise == 1 then
		encoder:add(nn.WhiteNoise(opt.noise_mean, opt.noise_std))
	end

	-- input is (nc) x 4 x 64 x 64
	encoder:add(nn.VolumetricConvolution(input_nc, ndf, 3, 4, 4, 1, 2, 2, 1, 1, 1)):add(nn.LeakyReLU(neg_val, true))

	-- input is (ndf) x 4 x 32 x 32
	encoder:add(nn.VolumetricConvolution(ndf, ndf*2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ndf*2)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ndf*2) x 2 x 16 x 16
	encoder:add(nn.VolumetricConvolution(ndf*2, ndf*4, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ndf*4)):add(nn.Dropout(0.5)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ndf*4) x 2 x 8 x 8
	encoder:add(nn.VolumetricConvolution(ndf*4, ndf*8, 4, 4, 4, 2, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ndf*8)):add(nn.Dropout(0.5)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ndf*8) x 1 x 4 x 4
	encoder:add(nn.VolumetricConvolution(ndf*8, ndf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ndf*8)):add(nn.Dropout(0.5)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ndf*8) x 1 x 2 x 2
	encoder:add(nn.VolumetricConvolution(ndf*8, ndf*8, 3, 4, 4, 1, 2, 2, 1, 1, 1))
	encoder:add(nn.VolumetricBatchNormalization(ndf*8)):add(nn.LeakyReLU(neg_val, true))
	
	-- input is (ndf*8) x 1 x 1 x 1
	encoder:add(nn.View(-1):setNumInputDims(4))
	
	-- input is (ndf*8)
	lstm:add(nn.LSTM(ndf*8, ndf*4))
	
	d_model:add(nn.Sequencer(encoder))
	d_model:add(nn.Sequencer(lstm))
	d_model:add(nn.Select(1, -1))
	d_model:add(nn.Linear(ndf*4, 1))
	d_model:add(nn.Sigmoid())

	return d_model
end
