import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
from Utils.Trainer import *

def print_means(means):
	for key in means:
		if key=='E':
			print("Mean [", key,"]=", means[key].numpy()*627.5096080305927, " kcal/mol.")
		elif key=='F':
			print("Mean [", key,"]=", means[key].numpy()*627.5096080305927/0.529177249, " kcal/mol/A.")
		else:
			#print("Mean [", key,"]=", means[key].numpy(), " Atomic unit.")
			#print("Mean [", key,"]=", means[key].numpy())
			print("{:5s}[{:2s}] = {:20.10f}".format('Mean',key,means[key].numpy()))

def print_acc(prop, name, title=None):
	for key in prop:
		#print(name,"[",key,"]=", prop[key].numpy())
		print("{:5s}[{:2s}] = {:20.10f}".format(name,key,prop[key].numpy()),end="")
	if title is not None:
		print(title)
	else:
		print("")

def print_all_acc(acc):
	if acc is None:
		return
	for keys in acc:
		if keys == 'Means':
			print_means(acc[keys])
		elif keys == 'R2':
			print_acc(acc[keys],"R2  "," = Coefficient of determination")
		elif keys == 'r2':
			print_acc(acc[keys],"r2  ", " = Pearson correlation coefficient squared")
		elif keys == 'Loss':
			print_acc(acc[keys],"Loss")
		else:
			print_acc(acc[keys],keys)
		#print("")

def validation_test(trainer, lossbest, best_checkpoint, best_loss_file, step):
	print("Validation data")
	print("---------------------")
	trainer.save_variable_backups()
	if trainer.use_average==1:
		trainer.optimizer.assign_average_vars(trainer.model.variables)
	means,loss=trainer.computeValidationAccuracy(verbose=False)
	if means is None:
		means,loss=trainer.computeTrainAccuracy(verbose=False)
	#means,loss=trainer.computeTrainAccuracy(verbose=False)
	print_means(means)
	st='Loss={:0.4}'.format(loss.numpy())
	if lossbest is not None:
		st = st +'\t ; Old best loss={:0.4}'.format(lossbest.numpy())
	print(st)
	if lossbest is None or loss.numpy()<lossbest.numpy():
		print("save chk new best loss=",loss.numpy())
		trainer.save_weights(fname=best_checkpoint)
		lossbest=loss
		props = {'Step':step,'Loss':loss.numpy()}
		#props.update(means)
		for key in means :
			props[key] = means[key].numpy()
		np.savez(best_loss_file, props=props)
	print("--------------------------------------------------------------")
	trainer.restore_variable_backups()
	return lossbest

def load_best_recorded_loss(best_loss_file):
	#save/load best recorded loss (only the best model is saved)
	print("Trying to load best recorded loss")
	if os.path.isfile(best_loss_file):
		loss_file   = np.load(best_loss_file,allow_pickle=True)
		props=loss_file['props'].item()
		print("Old result with best loss :", props)
		#print(props['Loss'])
		lossbest = tf.constant(props['Loss'])
		#[ print("best[", key,"]=", loss_file[key].item()) for key in props ]
	else:
		props = {'Step':0,'Loss':np.Inf,'E':np.Inf,'Q':np.Inf}
		np.savez(best_loss_file, props=props)
		lossbest=None
	return lossbest, props


def create_trainer(irModel, args):
	print("Creation of trainer")
	trainer=Trainer(irModel, args, dataFile=args.dataset)
	return  trainer

def add_title_to_metrics_file(f, acc):
	if acc is not None:
		f.write("# ")
		for p in acc:
			f.write('{:20s} '.format(p))
		f.write("\n")

def open_metrics_file(directory, fname):
	filename= os.path.join(directory, fname)
	if not os.path.exists(directory):
                os.makedirs(directory)
	f = open(filename,"a")
	return f,filename

def add_metrics_to_file(f, acc, addTitle=False):
	if acc is not None:
		if addTitle:
			add_title_to_metrics_file(f, acc)
		for p in acc:
			f.write('{:20.14e} '.format(acc[p].numpy()))
		f.write("\n")
		f.flush()

def open_validation_metrics_files(directory,  args):
	fnames = { 'mae':"mae.txt", 'ase':"ase.txt", 'rmse':"rmse.txt", 'R2':"R2.txt", 'r2':"r2.txt", 'Loss':"Loss.txt"}
	files = {}
	filenames = {}
	print("Opening validation metrics files")
	for key in fnames:
		files[key], filenames[key] = open_metrics_file(directory, fnames[key])
	return files,filenames

def add_metrics_to_files(files, acc, addTitle=False):
	if acc is not None:
		for key in acc:
			add_metrics_to_file(files[key], acc[key], addTitle=addTitle)

def close_metrics_files(files):
	for key in files:
		files[key].close()

def add_validation_metrics_to_files(trainer, metrics_files, addTitle=False):
	trainer.save_variable_backups()
	if trainer.use_average==1:
		trainer.optimizer.assign_average_vars(trainer.model.variables)
	acc=trainer.computeValidationAccuracies(verbose=True)
	add_metrics_to_files(metrics_files, acc, addTitle)
	trainer.restore_variable_backups()
	return acc

def open_train_metrics_files(directory,  args):
	fnames = { 'mae':"mae_train.txt", 'ase':"ase_train.txt", 'rmse':"rmse_train.txt", 'R2':"R2_train.txt", 'r2':"r2_train.txt", 'Loss':"Loss_train.txt"}
	files = {}
	filenames = {}
	print("Opening train metrics files")
	for key in fnames:
		files[key], filenames[key] = open_metrics_file(directory, fnames[key])
	return files,filenames

def open_test_metrics_files(directory,  args):
	fnames = { 'mae':"mae_test.txt", 'ase':"ase_test.txt", 'rmse':"rmse_test.txt", 'R2':"R2_test.txt", 'r2':"r2_test.txt", 'Loss':"Loss_test.txt"}
	files = {}
	filenames = {}
	print("Opening test metrics files")
	for key in fnames:
		files[key], filenames[key] = open_metrics_file(directory, fnames[key])
	return files,filenames

def add_train_metrics_to_files_from_sums(trainer, train_metrics_files, sums, addTitle=False):
	acc=trainer.computeAccuraciesFromSums(sums, verbose=False)
	add_metrics_to_files(train_metrics_files, acc, addTitle)

def add_train_metrics_to_files(trainer, metrics_files, addTitle=False):
	trainer.save_variable_backups()
	if trainer.use_average==1:
		trainer.optimizer.assign_average_vars(trainer.model.variables)
	acc=trainer.computeTrainAccuracies(verbose=True)
	add_metrics_to_files(metrics_files, acc, addTitle)
	trainer.restore_variable_backups()
	return acc

def add_test_metrics_to_files(trainer, metrics_files, addTitle=False):
	trainer.save_variable_backups()
	if trainer.use_average==1:
		trainer.optimizer.assign_average_vars(trainer.model.variables)
	acc=trainer.computeTestAccuracies(verbose=True)
	add_metrics_to_files(metrics_files, acc, addTitle)
	trainer.restore_variable_backups()
	return acc

def add_metrics_to_logs(writer_logs, acc, step, prefix=None):
	with writer_logs.as_default():
		if acc is not None:
			for keys in acc:
				for key in acc[keys]:
					if prefix is None:
						name=keys+"_"+key
					else:
						name=prefix+"_"+keys+"_"+key
					tf.summary.scalar(name, acc[keys][key], step=step)
