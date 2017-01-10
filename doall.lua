dofile 'config.lua'
color = require 'trepl.colorize'
dofile 'cudnn_convert_custom.lua'
dofile 'data.lua'
dofile 'model.lua'
dofile 'train.lua'
dofile 'validate.lua'

-- opt.manualSeed = torch.random(1, 10000)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

print(color.green'\n====> Loading data')
local train_data, valid_data, test_data = get_data(opt.data_path)
print(color.green'\n====> Data load complete')

print(color.blue"\nNumber of train examples " .. get_size(train_data))
print(color.blue"Number of validation examples " .. get_size(valid_data))
print(color.blue"Number of test examples " .. get_size(test_data))

print(color.green'\n====> Define generator model')
net_g = get_generator_u_model(opt.input_nc, opt.output_nc, opt.ngf)
net_g:apply(weights_init)
if opt.disp_model == 1 then 
	print(net_g)
end
print(color.blue"\nNumber of parameters in Generator : " .. net_g:getParameters():nElement())

print(color.green'\n====> Define discriminator model')
net_d = get_discriminator_model(opt.input_nc, opt.output_nc, opt.ndf)
net_d:apply(weights_init)
if opt.disp_model == 1 then
	print(net_d)
end
print(color.blue"\nNumber of parameters in Discriminator : " .. net_d:getParameters():nElement())

print(color.green"\n====> Define losses")
criterion = nn.BCECriterion()
criterionAE = nn.AbsCriterion()

print(color.green"\n====> Start training")
print(color.blue"\nBatch size : " .. opt.batch_size)
print(color.blue"Number of epochs : " .. opt.num_epochs .. "\n")
train(train_data[{{}, {1050}}], train_data, opt.num_epochs, opt.batch_size, valid_data)
