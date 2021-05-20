import pickle
from numpy.lib.function_base import iterable
import glob
import torch
import torchvision as tv
import pytorch_lightning as pl
from loguru import logger
from termcolor import colored
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from transformers.models.bert.modeling_bert import BertConfig

from model.oscar.modeling_bert import BertForImageCaptioning
from model.efficientnet.net import EfficientNet
from model.monocle import Monocle
from dataset.pl_bms import LitBBMS, LitDetBBMS
from dataset.tokenizer import basic_tokenizer
from dataset.collator import EncodedBatchCollator


# tk_model_file = "./checkpoints/bms_worldpiece_tokenizer.json"
tk_model_file = "./checkpoints/bms_tokenizer.json"


def test_visual_feat_extract():
    backbone = EfficientNet.from_pretrained(
        'efficientnet-b3',
        in_channels=3,
        include_top=False)
    print(backbone)

    x1 = torch.ones(1, 3, 320, 320)
    feat_map = backbone(x1)
    print(feat_map.shape)

    # x2 = torch.arange(0, 100).view(1, 1 ,10, 10).float()
    n = 10
    grid_x, grid_y = torch.meshgrid(torch.arange(n), torch.arange(n))
    x2 = torch.stack([grid_x, grid_y], dim=0).view(1, 2, n, n).float()
    boxes = [
        torch.tensor([[10, 10, 40, 40]]).float()
    ]
    rois = tv.ops.roi_align(x2, boxes, output_size=4, spatial_scale=.1, aligned=True)
    print(rois)


def test_base_dataset(batch_size=8, max_cap_len=None):
    from dataset import bms_caption, collator
    global tk_model_file
    train_dir = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val"
    labels = "/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv"

    # tokenizer = Tokenizer.from_file(tk_model_file)
    tokenizer = basic_tokenizer(custom_decoder=True)
    bbms_coll = collator.EncodedBatchCollator()
    bbms = bms_caption.EncodedBBMS(
        train_dir, labels, tokenizer,
        mlm=False, det_inchi=False, max_cap_len=max_cap_len, mask_prob=0.001)
    loader = DataLoader(bbms, batch_size=batch_size, collate_fn=bbms_coll, num_workers=0, shuffle=True)
    
    print(len(bbms))
    # print(bbms[1][2:])
    for batch_ndx, sample in enumerate(loader):
        # print(sample)
        if batch_ndx > 16: break
    return loader


def test_img_cap_bert():

    def to_cuda(stuff):
        if isinstance(stuff, torch.Tensor):
            return stuff.cuda()
        elif iterable(stuff):
            return [to_cuda(s) for s in stuff]
        else:
            raise RuntimeError(f"{type(stuff)} !?")
            
    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    pretrain_dir = "./config/bms_img_cap_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    # bert = BertForImageCaptioning.from_pretrained(pretrain_dir, config=config)
    bert = BertForImageCaptioning(config)
    bert.cuda()
    # print(bert)

    loader = test_base_dataset()
    sample = next(iter(loader))
    sample = [to_cuda(s) for s in sample[:8]]
    _, img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample
    fake_img_feat = torch.normal(0, 1, size=[img.shape[0], atten_mask.shape[-1] - ids.shape[-1], 1540])
    fake_img_feat = fake_img_feat.cuda()
    inputs = {
        'input_ids': ids, 'attention_mask': atten_mask,
        'token_type_ids': type_ids, 'img_feats': fake_img_feat, 
        'masked_pos': mask_pos, 'masked_ids': mask_ids,
        'is_decode': False,
    }
    bert.train()
    output = bert(**inputs)
    loss = output[0]
    loss.backward()
    print([o.shape for o in output[:2]])


def test_beam_serach_bert():

    def to_cuda(stuff):
        if isinstance(stuff, torch.Tensor):
            return stuff.cuda()
        elif isinstance(stuff, (list, tuple)):
            return [to_cuda(s) for s in stuff]
        else:
            return stuff
            
    pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    # pretrain_dir = "./config/bms_img_cap_bert/"
    max_length = 200
    config = BertConfig.from_pretrained(pretrain_dir)
    bert = BertForImageCaptioning.from_pretrained(pretrain_dir, config=config)
    # bert = BertForImageCaptioning(config)
    # bert.cuda()
    # bert = torch.jit.script(bert)
    # print(bert)

    loader = test_base_dataset(max_cap_len=max_length)
    tokenizer = loader.dataset.tokenizer
    sample = next(iter(loader))
    sample = [sample[0]] + [to_cuda(s) for s in sample[1:]]
    _, img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample[:8]
    fake_img_feat = torch.normal(0, 1, size=[img.shape[0], atten_mask.shape[-1] - ids.shape[-1], 1540])
    fake_img_feat = fake_img_feat.cuda()

    inputs = {
        'input_ids': ids,
        'attention_mask': atten_mask,
        'token_type_ids': type_ids,
        'img_feats': fake_img_feat, 
        'masked_pos': mask_pos,
        'is_decode': True,
        
        # hyperparameters of beam search
        'max_length': max_length,
        'num_beams': 3,
        "temperature": 1.0,
        "top_k": 3,
        "top_p": 1.0,
        "repetition_penalty": 1,
        "length_penalty": 1,
        "num_return_sequences": 1,
        "num_keep_best": 1,
        "od_labels_start_posid": -1,

        "bos_token_id": tokenizer.token_to_id("[CLS]"), 
        "pad_token_id": tokenizer.token_to_id("[PAD]"),
        "eos_token_ids": tokenizer.token_to_id("[SEP]"), 
        "mask_token_id": tokenizer.token_to_id("[MASK]"),
    }

    with open('batch.pickle', 'rb') as f:
        inputs = pickle.load(f)

        inputs['is_decode'] = True
        # inputs['is_training'] = False
        # del inputs['is_decode']
        # del inputs['do_sample']
    
    bert.eval()
    output = bert(**inputs)
    decoded, logprobs = output
    print(decoded)
    print(logprobs)
    print([o.shape for o in output[:2]])


def test_monocle(ckpt=None, is_training=True, device='cuda:1', max_length=None, beam_search=False,):
    """
    testing burt force token-wise inference
    """

    def to_cuda(stuff):
        if isinstance(stuff, torch.Tensor):
            return stuff.to(device)
        elif iterable(stuff):
            return [to_cuda(s) for s in stuff]
        else:
            raise RuntimeError(f"{type(stuff)} !?")
            
    global tk_model_file
    # tokenizer = Tokenizer.from_file(tk_model_file)
    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    tokenizer = basic_tokenizer(custom_decoder=True)
    # pretrain_dir = "./config/bms_img_cap_bert/"
    pretrain_dir = "./config/basic_tk_bms_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    if ckpt:
        bert = Monocle.load_from_checkpoint(ckpt, bert_config=config)
    else:
        bert = Monocle(config)
    bert.to(device)
    # print(bert)
    bert.eval()

    loader = test_base_dataset(batch_size=1, max_cap_len=max_length)
    
    for i, sample in enumerate(loader):
        # n = loader.dataset.imgids.index('4cf6b16ffa89')
        # sample = loader.dataset[n]
        # sample = EncodedBatchCollator()([sample])
        if i > 3: break
        sample_ids = sample[0]
        print(colored(sample_ids, color='blue'))
        sample = sample[1:]
        tokens = sample[-1]
        sample = [to_cuda(s) for s in sample[:7]]
        img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample
        ref_inchis = [''.join(t[1:]) for t in tokens]

        if beam_search:
            inputs = {
                'attention_mask': atten_mask,
                'token_type_ids': type_ids,
                'masked_pos': mask_pos,
                'is_decode': True,
                
                # hyperparameters of beam search
                'max_length': max_length,
                'num_beams': 5,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1,
                "repetition_penalty": 1,
                "length_penalty": 1,
                "num_return_sequences": 1,
                "num_keep_best": 1,
                "od_labels_start_posid": -1,

                "bos_token_id": tokenizer.token_to_id("[CLS]"), 
                "pad_token_id": tokenizer.token_to_id("[PAD]"),
                "eos_token_ids": [tokenizer.token_to_id("[SEP]")], 
                "mask_token_id": tokenizer.token_to_id("[MASK]"),
            }
        else:
            inputs = {
                'attention_mask': atten_mask,
                'token_type_ids': type_ids,
                'masked_pos': mask_pos,
                'masked_ids': mask_ids,
                'is_decode': False,
                'is_training': is_training,
            }
        output = bert(img, boxes, ids, **inputs)
        
        print(f'[{i}] seq_len: {ids.shape}, img size: {img.shape}, img_val_min_max: {img.min()}/{img.max()}, box: {[len(b) for b in boxes]}')
        input_strs = tokenizer.decode_batch(ids.cpu().numpy()[:, 1:])
        print(colored("input_strs: ", color='green'), input_strs)
        if not is_training:
            if beam_search:
                decoded = output[0]
                print('decoded: ', decoded)
                pred_strs = tokenizer.decode_batch(decoded.cpu().numpy()[:, -1])
                print(colored("decode pred_strs: ", color='green'), pred_strs)
            else:
                pred_ids = torch.argmax(output[0], dim=-1)
                pred_strs = tokenizer.decode_batch(pred_ids.cpu().numpy()[:, 1:])
                print(mask_pos)
                print(mask_ids)
                print(pred_ids[:, 1:])
                print(colored("pred_strs: ", color='green'), pred_strs)
            print(colored("ref_inchis: ", color='green'), ref_inchis)
        else:
            loss, logits = output[:2]
            print('loss: ', loss)
        print('-' * 100)


def test_monocle_overfit(ckpt=None, is_training=True, device='cuda:0', max_length=None, beam_search=False,):
    """
    testing burt force token-wise inference
    """

    def to_cuda(stuff):
        if isinstance(stuff, torch.Tensor):
            return stuff.to(device)
        elif iterable(stuff):
            return [to_cuda(s) for s in stuff]
        else:
            raise RuntimeError(f"{type(stuff)} !?")
            
    # pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    global tk_model_file
    tokenizer = Tokenizer.from_file(tk_model_file)
    pretrain_dir = "./config/bms_img_cap_bert/"
    config = BertConfig.from_pretrained(pretrain_dir)
    if ckpt:
        bert = Monocle.load_from_checkpoint(ckpt, bert_config=config)
    else:
        bert = Monocle(config)
    bert.to(device)
    # print(bert)
    bert.train()
    adam = bert.configure_optimizers()

    init_cnn_param = [p.cpu().detach() for p in bert.backbone.parameters()]

    loader = test_base_dataset(batch_size=8, max_cap_len=max_length)
    
    for i, sample in enumerate(loader):
        
        if i > 2:
            break
        sample = sample[1:]
        sample = [to_cuda(s) for s in sample]
        img, boxes, ids, type_ids, atten_mask, mask_pos, mask_ids = sample[:7]

        for j in range(128):
            adam.zero_grad()
            inputs = {
                'attention_mask': atten_mask,
                'token_type_ids': type_ids,
                'masked_pos': mask_pos,
                'masked_ids': mask_ids,
                'is_decode': False,
                'is_training': is_training,
            }
            output = bert(img, boxes, ids, **inputs)
            
            # print(f'[{i}] seq_len: {ids.shape}, img size: {img.shape}, img_val_min_max: {img.min()}/{img.max()}')
            loss, logits = output[:2]
            print(f'[{i},{j}] loss: ', loss)

            pred_ids = torch.argmax(logits, dim=-1)
            pred_strs = tokenizer.decode(pred_ids.cpu().numpy())
            token_delta = mask_ids[mask_ids!=0] - pred_ids

            loss.backward()
            adam.step()

            if loss < 0.02:
                param_d = [(p1 - p2.cpu().detach()).mean() for p1, p2 in zip(init_cnn_param, bert.backbone.parameters())]
                print(param_d)
            print('-' * 100)


def test_lit_data():
    global tk_model_file
    tokenizer = Tokenizer.from_file(tk_model_file)
    lit_dataset = LitDetBBMS(
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train',
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/val',
        tokenizer,
        '/home/ron/Downloads/bms-molecular-translation/bms-molecular-translation/train_labels.csv',
        num_worker=0,
        batch_size=8,
    )

    loader = lit_dataset.train_dataloader()
    for i, data in enumerate(loader):
        for b in range(len(data[0])):
            for d in data:
                print(d[b])
            if b > 2: break
        if i > 1: break


def transfer_coco_cap_weight():
    pretrain_dir = "/home/ron/Downloads/coco_captioning_base_scst/checkpoint-15-66405"
    config = BertConfig.from_pretrained(pretrain_dir)
    bert = BertForImageCaptioning.from_pretrained(pretrain_dir, config=config)

    pretrain_dir = "./config/basic_tk_bms_bert/"
    ckpt = "/home/ron/Projects/MolecularTranslation/checkpoints/dev/lightning_logs/version_8/checkpoints/epoch=0-step=103332.ckpt"
    config = BertConfig.from_pretrained(pretrain_dir)
    monocle = Monocle(config)
    mono_state = monocle.state_dict()
    ckpt_state = torch.load(ckpt)['state_dict']
    match_state = {
        k: (ckpt_state[k] if ckpt_state[k].shape == mono_state[k].shape else mono_state[k])
        for k in mono_state.keys()
    }
    # monocle = Monocle.load_from_checkpoint(ckpt, bert_config=config)
    monocle.load_state_dict(match_state)

    coco_state = bert.state_dict()
    mono_state = monocle.bert.state_dict()
    assert len(coco_state) == len(mono_state), \
        f"{[k for k in coco_state.keys() if k not in mono_state]}"
    match_state = {
        k: (coco_state[k] if coco_state[k].shape == mono_state[k].shape else mono_state[k])
        for k in mono_state.keys()
    }
    monocle.bert.load_state_dict(match_state, strict=True)
    torch.save(monocle.state_dict(), "checkpoints/coco_monocle.pth")


def det_inchi_test(test_dir):
    glob.glob()


with logger.catch(reraise=True):
    # test_visual_feat_extract()
    # test_base_dataset()
    # test_img_cap_bert()
    # test_monocle(
    #     is_training=False,
    #     beam_search=True,
    #     max_length=200,
    #     ckpt="/home/ron/Projects/MolecularTranslation/checkpoints/dev/lightning_logs/version_16/checkpoints/epoch=0-step=50632.ckpt",
    #     device='cuda:0'
    # )
    # test_monocle()

    test_beam_serach_bert()

    # test_lit_data()
    # test_monocle_overfit()
    # transfer_coco_cap_weight()