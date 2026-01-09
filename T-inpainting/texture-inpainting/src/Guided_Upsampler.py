import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel, InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
import cv2
import torchvision.utils as vutils

class Guided_Upsampler():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'

        self.debug = False
        self.model_name = model_name

        self.edge_model = EdgeModel(config).to(config.DEVICE)

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()
        elif self.config.MODEL == 2:
            self.inpaint_model.load()
        else:
            self.edge_model.load()
            self.inpaint_model.load()
        
    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()
        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model.save()
        else:
            self.edge_model.save()
            self.inpaint_model.save()

    def train(self):
        # 先检查训练数据集是否为空
        total = len(self.train_dataset)
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
            
        # 检查批次大小是否合适
        if self.config.BATCH_SIZE > total:
            print(f'警告: 批次大小 ({self.config.BATCH_SIZE}) 大于数据集大小 ({total})。将批次大小调整为 {total}。')
            actual_batch_size = min(self.config.BATCH_SIZE, total)
        else:
            actual_batch_size = self.config.BATCH_SIZE
            
        # 创建数据加载器
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=actual_batch_size,
            num_workers=0,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        
        # 计算数据集的batch数量和预计的最大epoch数
        batches_per_epoch = len(train_loader)
        
        # 防止除以零错误
        if batches_per_epoch == 0:
            print('警告：每个epoch的批次数为0，这可能是因为批次大小设置不当或数据集加载有问题！')
            print(f'训练数据集路径: {self.config.TRAIN_FLIST}')
            print(f'训练边缘数据集路径: {self.config.TRAIN_EDGE_FLIST}')
            print(f'训练掩码数据集路径: {self.config.TRAIN_MASK_FLIST}')
            batches_per_epoch = 1  # 设置为1以防止除以零错误
            
        max_epoch = max_iteration // batches_per_epoch + 1
        
        # 获取当前迭代次数
        if model == 1:
            current_iteration = self.edge_model.iteration
        elif model == 2 or model == 3:
            current_iteration = self.inpaint_model.iteration
        else:
            current_iteration = self.inpaint_model.iteration
            
        # 计算当前应该处于哪个epoch
        current_epoch = current_iteration // batches_per_epoch if batches_per_epoch > 0 else 0

        print(f'数据集大小: {total}张图像')
        print(f'批次大小: {actual_batch_size}')
        print(f'每个epoch的批次数: {batches_per_epoch}')
        print(f'最大迭代次数: {max_iteration}')
        print(f'预计最大epoch数: {max_epoch}')
        print(f'当前迭代次数: {current_iteration}')
        print(f'当前epoch: {current_epoch}')

        keep_training = True
        epoch = current_epoch

        while(keep_training and epoch < max_epoch):
            epoch += 1
            print(f'\n\nTraining epoch: {epoch}/{max_epoch}')

            if self.config.No_Bar:
                pass
            else:
                progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            # 跟踪当前epoch内的批次计数
            batch_num = 0
            
            for items in train_loader:
                batch_num += 1
                self.edge_model.train()
                self.inpaint_model.train()
                images, structure, images_gray, edges, masks = self.cuda(*items)

                # edge model
                if model == 1:
                    outputs, gray, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)
                    gray = (gray * masks) + (images_gray * (1 - masks))

                    # metrics
                    precision, recall = self.edgeacc(edges * masks, outputs * masks)
                    psnr = self.psnr(self.postprocess(images_gray), self.postprocess(gray))
                    mae = (torch.sum(torch.abs(images_gray - gray)) / torch.sum(images_gray)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))
                    logs.append(('precision', precision))
                    logs.append(('recall', recall.item()))

                    # backward
                    # self.edge_model.backward(gen_loss, dis_loss)
                    iteration = self.edge_model.iteration

                elif model == 2:
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, structure, edges, images_gray, masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    # self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                # inpaint with edge model
                elif model == 3:
                    outputs, gray = self.edge_model(images_gray, edges, masks)
                    outputs = outputs * masks + edges * (1 - masks)
                    gray = gray * masks + images_gray * (1 - masks)
                    outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, structure, outputs.detach(), gray.detach(), masks)
                    outputs_merged = (outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    # self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                # joint model
                else:
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, edges, masks)
                    e_outputs = e_outputs * masks + edges * (1 - masks)

                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.inpaint_model.process(images, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (images * (1 - masks))

                    # metrics
                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                    precision, recall = self.edgeacc(edges * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.edge_model.backward(e_gen_loss, e_dis_loss)
                    iteration = self.inpaint_model.iteration

                # 添加批次进度信息
                epoch_progress = f"{batch_num}/{batches_per_epoch}"

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", f"{epoch}/{max_epoch}"),
                    ("batch", epoch_progress),
                    ("iter", iteration),
                ] + logs

                if self.config.No_Bar:
                    pass
                else:
                    progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=False
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.inpaint_model.eval()

        if self.config.No_Bar:
            pass
        else:
            progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, structure, images_gray, edges, masks = self.cuda(*items)

            if model == 1:
                # train
                outputs, gray, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)
                gray = (gray * masks) + (images_gray * (1 - masks))

                # metrics
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                psnr = self.psnr(self.postprocess(images_gray), self.postprocess(gray))
                mae = (torch.sum(torch.abs(images_gray - gray)) / torch.sum(images_gray)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))

            # inpaint model
            elif model == 2:
                # eval
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, structure, edges, images_gray, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            elif model == 3:
                # train
                outputs, gray = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)
                gray = gray * masks + images_gray * (1 - masks)

                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, structure, outputs.detach(),
                                                                               gray.detach(), masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

            logs = [("it", iteration), ] + logs
            if self.config.No_Bar:
                pass
            else:
                progbar.add(len(images), values=logs)

    def test(self):
        self.edge_model.eval()
        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.test_batch_size,
        )

        index = 0
        for items in test_loader:
            print(index)
            name = self.test_dataset.load_name(index)
            
            print(name)
            
            if self.config.same_face:
                path = os.path.join(self.results_path, name)
            else:
                # path = os.path.join(self.results_path, name[:-4]+"_%d"%(index%self.config.condition_num)+'.png')
                path = os.path.join(self.results_path, name)
            images, structure, images_gray, edges, masks = self.cuda(*items)

            # images, images_gray,  masks = self.cuda(*items)
            index += self.config.test_batch_size

            # inpaint model
            if model == 1:
                # train
                outputs, gray, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks)
                gray = (gray * masks) + (images_gray * (1 - masks))
                outputs_merged = outputs * masks + edges * (1 - masks)


            elif model == 2:

                outputs = self.inpaint_model(images, structure, edges, images_gray, masks)
                if self.config.merge:
                    outputs_merged = (outputs * masks) + (images * (1 - masks))
                else:
                    outputs_merged = outputs

                fname, fext = name.split('.')
                grays = self.postprocess(images_gray)[0]
                # print(index, name+"_gray")
                imsave(grays, os.path.join(self.results_path, fname + '_gray.' + fext))
                # masked = self.postprocess(images * (1 - masks))[0]
                # imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))
                edges = self.postprocess(edges)[0]
                # print(index, "edge"name+)
                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))



            elif model == 3:
                # train
                outputs, gray = self.edge_model(images_gray, edges, masks)
                outputs = outputs * masks + edges * (1 - masks)
                gray = gray * masks + images_gray * (1 - masks)
                print("gray:",gray.shape)
                print("outputs:",outputs.shape)

                fname, fext = name.split('.')
                # grays = self.postprocess(gray)[0]
                # print(index, name+"_gray")
                # imsave(grays, os.path.join(self.results_path, fname + '_gray.' + fext))
                # masked = self.postprocess(images * (1 - masks))[0]
                # imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))
                # output = self.postprocess(outputs)[0]
                # print(index, "edge"name+)
                # imsave(output, os.path.join(self.results_path, fname + '_edge.' + fext))

                outputs,_,_,_= self.inpaint_model.process(images, structure, outputs.detach(), gray.detach(), masks)
                print("!!:",outputs.shape)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

            if self.config.same_face:
                all_tensor=[images,edges,images * (1 - masks),outputs_merged]
                all_tensor=torch.cat(all_tensor,dim=0)
                vutils.save_image(all_tensor,path,nrow=self.config.test_batch_size,padding=0,normalize=False)
                print(index, name)
            else:
                fname, fext = name.split('.')
                output1 = self.postprocess(outputs_merged)
                print("%%:",output1.shape)
                # # print(index, "edge"name+)
                # imsave(output, os.path.join(self.results_path, fname + '_inpainting.' + fext))
                imsave(output1, os.path.join(self.results_path, fname + '_inpainting.' + fext))
                # gray = self.postprocess(gray)[0]
                # # print(index, name+"_gray")
                # imsave(gray, os.path.join(self.results_path, fname + '_gray.' + fext))
                # masked = self.postprocess(images * (1 - masks))[0]
                # imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, structure, images_gray, edges, masks = self.cuda(*items)

        if model == 1:
            # train
            inputs = (images_gray * (1 - masks)) + masks
            iteration = self.edge_model.iteration
            outputs, gray = self.edge_model(images_gray, edges, masks)
            outputs_merged = outputs * masks + edges * (1 - masks)
            gray_merged=gray * masks + images_gray * (1 - masks)

        elif model == 2:
            iteration = self.inpaint_model.iteration
            outputs = self.inpaint_model(images, structure, edges, images_gray, masks)
            inputs = (images * (1 - masks)) + masks
            outputs_merged = (outputs * masks) + (images * (1 - masks))

        elif model == 3:
            # train
            inputs = (images * (1 - masks)) + masks
            iteration = self.inpaint_model.iteration
            outputs, gray = self.edge_model(images_gray, edges, masks)
            outputs = outputs * masks + edges * (1 - masks)
            gray = gray * masks + images_gray * (1 - masks)

            outputs = self.inpaint_model(images, structure, outputs.detach(), gray.detach(), masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))


        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + "_edge.png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()