import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
from tqdm import tqdm
import os
import json

class Trainer(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self, config, model, optimizer, val_dataloader, ss_dataloader, writer, cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10

		self.config = config
		self.model = model
		self.optimizer = optimizer
		self.train_dataloader = None
		self.val_dataloader = val_dataloader
		self.ss_dataloader = ss_dataloader
		self.writer = writer
		# self.len_model_parameters = len(list(self.model.parameters()))

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		self.model.train()

		# Train loop
		for data in tqdm(self.train_dataloader):

			# efficiently zero gradients
			# for p in self.model.parameters():
			# 	p.grad = None

			loss, _ = self.step(data)
			self.optimizer.zero_grad()
			try:
				loss.backward()
			except:
				print(data)
			loss_epoch += float(loss.item())

			num_batches += 1
			self.optimizer.step()

			self.writer.add_scalar('train_loss', loss.item(), self.cur_iter)
			self.cur_iter += 1
		
		
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

	def validate(self):
		self.model.eval()

		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.

			# Validation loop
			for batch_num, data in enumerate(tqdm(self.val_dataloader), 0):
				
				wp_epoch += float(self.step(data)[0])
				num_batches += 1
					
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Wp: {wp_loss:3.3f}')

			self.writer.add_scalar('val_loss', wp_loss, self.cur_epoch)
			self.val_loss.append(wp_loss)

	def save(self):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		
		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
		}

		# Save the recent model/optimizer states
		torch.save(self.model.state_dict(), os.path.join(self.config['logdir'], 'model.pth'))
		torch.save(self.optimizer.state_dict(), os.path.join(self.config['logdir'], 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		with open(os.path.join(self.config['logdir'], 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(self.model.state_dict(), os.path.join(self.config['logdir'], 'best_model.pth'))
			torch.save(self.optimizer.state_dict(), os.path.join(self.config['logdir'], 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')	

	def get_labels(self):
		pass

	def step(self, data):
		pass

	def loss_fn(self, pred, gt):
		if self.config['predict_confidence']:
			bbl = 1
			x = bbl - torch.minimum(bbl*torch.ones_like(pred[:,:,0]), torch.abs(pred[:,:,0]-gt[:,:,0]))
			y = bbl - torch.minimum(bbl*torch.ones_like(pred[:,:,1]), torch.abs(pred[:,:,1]-gt[:,:,1]))

			ia = x*y
			ua = (2*bbl) - ia
			iou = ia/ua

			return F.l1_loss(pred[:,:,:2], gt).mean() + F.l1_loss(pred[:,:,2], iou).mean()
		else:
			return F.l1_loss(pred, gt, reduction='none').mean()



def step(self, data):
	
	# create batch and move to GPU
	fronts_in = data['fronts']
	fronts = []
	for i in range(self.config['seq_len']):
		fronts.append(fronts_in[i].to(self.config['device'], dtype=torch.float32))

	# target point
	# gt_velocity = data['velocity'].to(self.config['device'], dtype=torch.float32)
	target_point = torch.stack(data['target_point'], dim=1).to(self.config['device'], dtype=torch.float32)
	nav_command = data['nav_command'].to(self.config['device'])
	
	v1 = data['v1'].reshape((-1,1)).to(self.config['device'], dtype=torch.float32)
	v2 = data['v1'].reshape((-1,1)).to(self.config['device'], dtype=torch.float32)
	
	# inference
	encoding = [self.model.image_encoder(fronts)]
	
	pred_wp = self.model(feature_emb=encoding, v1=v1, v2=v2, target_point=target_point, nav_command=nav_command)
	
	gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(self.config['device'], dtype=torch.float32) for i in range(self.config['seq_len'], len(data['waypoints']))]
	gt_waypoints = torch.stack(gt_waypoints, dim=1).to(self.config['device'], dtype=torch.float32)
	

	# if not torch.isfinite(target_point).all(): print('target_point', target_point)
	# if not torch.isfinite(command).all(): print('command', command)
	# if not torch.isfinite(v1).all(): print('v1', v1)
	# if not torch.isfinite(v2).all(): print('v2', v2)
	# for front in fronts:
	# 	if not torch.isfinite(front).all(): print('front', front)
	# for enc in encoding:
	# 	if not torch.isfinite(enc).all(): print('enc', enc)
	# if not torch.isfinite(pred_wp).all(): print('pred_wp', pred_wp)
	# if not torch.isfinite(gt_waypoints).all(): print('gt_waypoints', gt_waypoints)

	# loss = (1+sum(torch.linalg.norm(p.flatten(), 1) for p in list(self.model.parameters())[(self.len_model_parameters//10):-(self.len_model_parameters//10)])) * F.l1_loss(pred_wp, gt_waypoints, reduction='none').mean()
	loss = self.loss_fn(pred_wp, gt_waypoints)
	return loss, [data['scene'], v1, v2, target_point, nav_command, pred_wp, gt_waypoints]



def get_labels(self):
	self.model.eval()
	pseudo_data_path = self.config['pseudo_data']

	data_dict = dict()
	with torch.no_grad():	

		# Validation loop
		for data in tqdm(self.ss_dataloader):
			
			scenes, fronts, v1s, v2s, target_points, nav_commands, pred_wp, gt_waypoints = self.step(data)[1]
			for i in range(len(scenes)):
				
				if self.config['predict_confidence']:
					if pred_wp[i][len(pred_wp[0])-1][2].item() < self.config['confidence_threshold']:
						continue

				data_dict[scenes[i]] = dict(
                    v2 = v2s[i].item(),
                    v1 = v1s[i].item(),
                    scene=scenes[i],
                    target_point=target_points[i].item(),
					waypoints=[],
					nav_command=nav_commands[i].cpu(),
                )

				data_dict[scenes[i]]['waypoints'].append((0.0,0.0))
				for j in range(len(pred_wp[0])):
					data_dict[scenes[i]]['waypoints'].append((pred_wp[i][j][0].item(), pred_wp[i][j][1].item()))
				
				if self.config['predict_confidence']:
					data_dict[scenes[i]]['confidence'] = pred_wp[i][len(pred_wp[0])-1][2].item()
    
	np.save(pseudo_data_path, list(data_dict.values()))

