# from train import *

# Config
config = GlobalConfig(self_supervised_training=args.sst, ssd_dir=args.ssd_dir)

# Data
# train_set = CARLA_Data(root=config.train_data, config=config)
ssd_set = CARLA_Data(root=config.ssd_data, config=config, is_imgaug=False)
val_set = CARLA_Data(root=config.val_data, config=config, is_imgaug=False)

# dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataloader_ssd = DataLoader(ssd_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Model
model = AIM(config, args.device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
trainer = Engine(config=config, model=model, optimizer=optimizer, val_dataloader=dataloader_val, ss_dataloader=dataloader_ssd)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)

if args.load_model:
	try:
		model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
		print("model loaded")
	except:
		raise 'failed to load the model'


if config.self_supervised_training:
	print("Training with Pseudolabels")
	train_set = CARLA_Data(root=(config.ssd_train_data+config.train_data), config=config, is_imgaug=True)
	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
	trainer.train_dataloader = dataloader_train

	for epoch in range(trainer.cur_epoch, args.epochs): 
		trainer.train()
		if epoch % args.val_every == 0: 
			trainer.validate()
			trainer.save()
	print("Collect Labels")
	trainer.get_labels()
    # print("Fine Tuning")
	# train_set = CARLA_Data(root=config.train_data, config=config)
	# dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
	# for epoch in range(trainer.cur_epoch, args.epochs): 
	# 	trainer.train()
	# 	if epoch % args.val_every == 0: 
	# 		trainer.validate()
	# 		trainer.save()

if not config.self_supervised_training:
	print("Supervised Training")
	train_set = CARLA_Data(root=config.train_data, config=config, is_imgaug=True)
	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
	trainer.train_dataloader = dataloader_train

	for epoch in range(trainer.cur_epoch, args.epochs): 
		trainer.train()
		if epoch % args.val_every == 0: 
			trainer.validate()
			trainer.save()
	print("Collect Labels")
	trainer.get_labels()