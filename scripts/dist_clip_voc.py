import os
# 在这里添加下面这行代码
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import logging
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_model.model_attn_aff_voc import WeCLIP


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/your/path/WeCLIP/configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None):

    preds, gts, cams, aff_gts = [], [], [], []
    num = 1
    seg_hist = np.zeros((21, 21))
    cam_hist = np.zeros((21, 21))
    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        segs, cam, attn_loss, mamba_class = model(inputs, name, 'val')

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

        num+=1

        if num % 1000 ==0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    model.train()
    return seg_score, cam_score

def validate(model=None, data_loader=None, cfg=None):

    preds, gts, cams = [], [], []
    cls_labels = []
    mamba_preds = []  # 新增：存储 mamba_class 的预测结果

    num = 1
    seg_hist = np.zeros((21, 21))
    cam_hist = np.zeros((21, 21))

    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        cls_label = cls_label.cuda()

        segs, cam, attn_loss, mamba_class = model(inputs, name, 'val')

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

        cls_labels.append(cls_label.detach().cpu())
        mamba_preds.append(mamba_class.detach().cpu())  # 新增：记录 mamba_class

        num += 1

        if num % 1000 == 0:
            seg_hist, _ = evaluate.scores(gts, preds, seg_hist)
            cam_hist, _ = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams = [], [], []

    # 最后一批
    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)

    # 合并所有的分类预测与标签
    cls_labels = torch.cat(cls_labels, dim=0)
    
    # 新增：合并所有的 mamba_class 预测结果
    mamba_preds = torch.cat(mamba_preds, dim=0)
    mamba_cls_metrics = multilabel_accuracy_metrics(mamba_preds, cls_labels)  # 新增：计算 mamba_cls_metrics

    model.train()
    return seg_score, cam_score, mamba_cls_metrics  # 修改返回值


def multilabel_accuracy_metrics(cls_token, cls_labels, threshold=0.5):
    """
    计算多标签分类的准确率和F1分数等指标（含中文注释）

    参数:
        cls_token: 模型输出的logits，形状为 [batch_size, num_labels]
        cls_labels: 真实标签，形状为 [batch_size, num_labels]，元素为0或1
        threshold: 判断为正样本的阈值（对sigmoid输出进行二值化），默认0.5

    返回:
        一个字典，包含多标签分类常用评估指标（含中文解释）
    """
    # 将sigmoid输出转为0/1预测标签
    preds = (torch.sigmoid(cls_token) >= threshold).int()
    labels = cls_labels.int()

    # === 1. 严格匹配准确率（Exact Match Accuracy）===
    # 每个样本的预测标签必须完全与真实标签一致才算正确
    exact_match = (preds == labels).all(dim=1).float().mean().item()

    # === 2. 样本级准确率（Sample-based Accuracy / Jaccard Index）===
    # 每个样本计算预测标签与真实标签的交并比，然后对所有样本取平均
    intersection = (preds & labels).sum(dim=1).float()
    union = (preds | labels).sum(dim=1).float()
    jaccard = torch.where(union == 0, torch.ones_like(union), intersection / union)
    sample_accuracy = jaccard.mean().item()

    # === 3. 微平均指标（Micro Precision / Recall / F1 / Accuracy）===
    # 全局计算TP/FP/FN，适合类别不均衡
    TP = (preds & labels).sum().float()
    FP = (preds & (1 - labels)).sum().float()
    FN = ((1 - preds) & labels).sum().float()
    precision_micro = TP / (TP + FP + 1e-8)
    recall_micro = TP / (TP + FN + 1e-8)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8)
    micro_accuracy = TP / (TP + FP + FN + 1e-8)

    # === 4. 宏平均指标（Macro Precision / Recall / F1 / Accuracy）===
    # 每个标签（列）分别计算，再取平均，适合分析各个标签表现
    TP_per_label = (preds & labels).sum(dim=0).float()
    FP_per_label = (preds & (1 - labels)).sum(dim=0).float()
    FN_per_label = ((1 - preds) & labels).sum(dim=0).float()

    precision_macro = TP_per_label / (TP_per_label + FP_per_label + 1e-8)
    recall_macro = TP_per_label / (TP_per_label + FN_per_label + 1e-8)
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro + 1e-8)
    f1_macro_score = f1_macro.mean().item()

    # 宏平均准确率：每列的预测正确率的平均
    correct_per_label = (preds == labels).float().mean(dim=0)
    macro_accuracy = correct_per_label.mean().item()

    return {
        "exact_match_accuracy": exact_match,  # 严格匹配准确率
        "sample_accuracy": sample_accuracy,  # 样本级准确率（交并比）
        "micro_accuracy": micro_accuracy.item(),  # 微平均准确率（TP / 全部预测和真实的并集）
        "micro_f1": f1_micro.item(),  # 微平均F1分数（适合类别不平衡）
        "macro_f1": f1_macro_score,  # 宏平均F1分数（适合分析所有标签整体表现）
        "macro_accuracy": macro_accuracy  # 宏平均准确率（每列准确率取平均）
    }


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask



def train(cfg):

    num_workers = 10
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='train',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4,
                              worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False,
                            worker_init_fn=worker_init_fn)

    WeCLIP_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda',
        vMConfig=cfg.VMMODEL
    )
    logging.info('\nNetwork config: \n%s'%(WeCLIP_model))
    param_groups = WeCLIP_model.get_param_groups()
    WeCLIP_model.cuda()


    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    bast_acore = 0

    for n_iter in range(cfg.train.max_iters):
        
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        segs, cam, attn_pred, mamba_class = WeCLIP_model(inputs.cuda(), img_name)

        pseudo_label = cam

        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        fts_cam = cam.clone()

            
        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        mamba_cls_loss  = F.binary_cross_entropy_with_logits(mamba_class,cls_labels.cuda())

        loss = 1 * seg_loss + 0.1*attn_loss + 1 * mamba_cls_loss


        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item(), 'mamba_cls_loss': mamba_cls_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size


            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, mamba_cls_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'), avg_meter.pop('mamba_cls_loss'), seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item(), "mamba_cls_loss": mamba_cls_loss.item()}, global_step=n_iter)

        
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "WeCLIP_model_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            if (n_iter + 1) > 26000:
                torch.save(WeCLIP_model.state_dict(), ckpt_name)
            seg_score, cam_score, mamba_cls_metrics = validate(model=WeCLIP_model, data_loader=val_loader, cfg=cfg)
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)
            logging.info("mamba cls metrics:")
            logging.info(mamba_cls_metrics)

            if bast_acore < seg_score["miou"]:
                bast_acore = seg_score["miou"]
                logging.info("bast acore:")
                logging.info(bast_acore)
            
    logging.info("bast acore:")
    logging.info(bast_acore)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)
