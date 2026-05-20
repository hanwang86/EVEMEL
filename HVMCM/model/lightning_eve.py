import gc
import math

import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from model.modeling_eve import BiGIM, HVMCMEncoder
from utils.loss import HardNegativeContrastiveLoss, IMNContrastiveLoss


class LightningForEVEMEL(pl.LightningModule):
    def __init__(self, args):
        super(LightningForEVEMEL, self).__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.encoder = HVMCMEncoder(args)
        self.timm = BiGIM(cls_dim=self.encoder.embed_dim, tok_dim=self.encoder.text_dim)
        self.iimm = BiGIM(cls_dim=self.encoder.embed_dim, tok_dim=self.encoder.vision_dim)

        self.loss_fct = HardNegativeContrastiveLoss()
        self.imn_loss_fct = IMNContrastiveLoss()

    def training_step(self, batch, batch_idx):
        ent_batch = {}
        mention_batch = {}
        neg_batch = {}

        for k, v in batch.items():
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            elif k.startswith('neg_'):
                neg_batch[k.replace('neg_', '')] = v
            else:
                mention_batch[k] = v

        if 'empty_img_flag' in ent_batch:
            ent_batch.pop('empty_img_flag')

        men_fused, men_t_cls, men_v_cls, men_t_tok, men_v_tok, men_router_weights = self.encoder(**mention_batch)
        ent_fused, ent_t_cls, ent_v_cls, ent_t_tok, ent_v_tok, ent_router_weights = self.encoder(**ent_batch)

        neg_fused, neg_t_cls, neg_v_cls, neg_t_tok, neg_v_tok = None, None, None, None, None
        if len(neg_batch) > 0:
            neg_fused, neg_t_cls, neg_v_cls, neg_t_tok, neg_v_tok, _ = self.encoder(**neg_batch)

        loss_main = self.loss_fct(men_fused, ent_fused, neg_fused)
        score_timm = self.timm(men_t_cls, men_t_tok, ent_t_cls, ent_t_tok)
        score_iimm = self.iimm(men_v_cls, men_v_tok, ent_v_cls, ent_v_tok)

        neg_score_timm, neg_score_iimm = None, None
        if neg_fused is not None:
            neg_score_timm = self.timm(men_t_cls, men_t_tok, neg_t_cls, neg_t_tok)
            neg_score_iimm = self.iimm(men_v_cls, men_v_tok, neg_v_cls, neg_v_tok)

        loss_timm = self.imn_loss_fct(score_timm, neg_score_timm)
        loss_iimm = self.imn_loss_fct(score_iimm, neg_score_iimm)

        mean_weights = men_router_weights.mean(dim=(0, 1))
        w_text = mean_weights[0]
        w_vis_sum = mean_weights[1:].sum()
        dynamic_scale_text = torch.clamp(w_text, min=0.1, max=1).detach()
        dynamic_scale_vis = torch.clamp(w_vis_sum, min=0.05, max=0.5).detach()

        loss = loss_main + dynamic_scale_text * loss_timm + dynamic_scale_vis * loss_iimm

        self.log('Train/dyn_text', dynamic_scale_text, on_step=True, prog_bar=True)
        self.log('Train/dyn_vis', dynamic_scale_vis, on_step=True, prog_bar=True)
        self.log('Train/loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_start(self):
        if hasattr(self, 'ent_t_tok_db') and self.ent_t_tok_db is not None:
            self.ent_fused_db = self.ent_t_cls_db = self.ent_v_cls_db = None
            self.ent_t_tok_db = self.ent_v_tok_db = None
            gc.collect()

        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        out_fused, out_t_cls, out_v_cls, out_t_tok, out_v_tok = [], [], [], [], []

        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc='UpdateEmbed', total=len(entity_dataloader)):
                batch = pl.utilities.move_data_to_device(batch, self.device)
                if 'empty_img_flag' in batch:
                    batch.pop('empty_img_flag')
                if 'sample_type' in batch:
                    batch.pop('sample_type')

                ent_fused, ent_t_cls, ent_v_cls, ent_t_tok, ent_v_tok, _ = self.encoder(**batch)
                out_fused.append(ent_fused.cpu().half())
                out_t_cls.append(ent_t_cls.cpu().half())
                out_v_cls.append(ent_v_cls.cpu().half())
                out_t_tok.append(ent_t_tok.cpu().half())
                out_v_tok.append(ent_v_tok.cpu().half())

        self.ent_fused_db = torch.concat(out_fused, dim=0)
        self.ent_t_cls_db = torch.concat(out_t_cls, dim=0)
        self.ent_v_cls_db = torch.concat(out_v_cls, dim=0)
        self.ent_t_tok_db = torch.concat(out_t_tok, dim=0)
        self.ent_v_tok_db = torch.concat(out_v_tok, dim=0)

        del out_fused, out_t_cls, out_v_cls, out_t_tok, out_v_tok
        gc.collect()

    def validation_step(self, batch, batch_idx):
        answer = batch.pop('answer')
        batch_size = len(answer)

        men_fused, men_t_cls, men_v_cls, men_t_tok, men_v_tok, men_router_weights = self.encoder(**batch)

        scores = []
        chunk_size = self.args.data.eval_chunk_size

        mean_weights = men_router_weights.mean(dim=(0, 1))
        w_text = mean_weights[0]
        w_vis_sum = mean_weights[1:].sum()
        dynamic_scale_text = torch.clamp(w_text, min=0.1, max=1).detach()
        dynamic_scale_vis = torch.clamp(w_vis_sum, min=0.05, max=0.5).detach()

        for idx in range(math.ceil(self.args.data.num_entity / chunk_size)):
            start_pos = idx * chunk_size
            end_pos = (idx + 1) * chunk_size

            chunk_fused = self.ent_fused_db[start_pos:end_pos].to(self.device).float()
            chunk_t_cls = self.ent_t_cls_db[start_pos:end_pos].to(self.device).float()
            chunk_v_cls = self.ent_v_cls_db[start_pos:end_pos].to(self.device).float()
            chunk_t_tok = self.ent_t_tok_db[start_pos:end_pos].to(self.device).float()
            chunk_v_tok = self.ent_v_tok_db[start_pos:end_pos].to(self.device).float()

            score_hv = torch.matmul(men_fused, chunk_fused.t())
            score_timm = self.timm(men_t_cls, men_t_tok, chunk_t_cls, chunk_t_tok)
            score_iimm = self.iimm(men_v_cls, men_v_tok, chunk_v_cls, chunk_v_tok)
            chunk_score = score_hv + dynamic_scale_text * score_timm + dynamic_scale_vis * score_iimm
            scores.append(chunk_score)

        scores = torch.concat(scores, dim=-1)
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()
        return dict(rank=tgt_rank)

    def _eval_epoch_end(self, outputs, stage_name, log_prefix):
        self.entity_fused_embeds = None
        self.ent_fused_db = None
        self.ent_t_cls_db = None
        self.ent_v_cls_db = None
        self.ent_t_tok_db = None
        self.ent_v_tok_db = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ranks = np.concatenate([_['rank'] for _ in outputs])
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()
        mr = ranks.mean()
        mrr = (1. / ranks).mean()

        print(f"\n{'=' * 20} {stage_name} Epoch {self.current_epoch} {'=' * 20}")
        print(f"Hits@1 : {hits1:.4f}")
        print(f"Hits@3 : {hits3:.4f}")
        print(f"Hits@5 : {hits5:.4f}")
        print(f"Hits@10: {hits10:.4f}")
        print(f"Hits@20: {hits20:.4f}")
        print(f"MRR    : {mrr:.4f}")
        print(f"MR     : {mr:.4f}")
        print(f"{'=' * 60}\n")

        self.log(f"{log_prefix}/hits20", hits20)
        self.log(f"{log_prefix}/hits10", hits10)
        self.log(f"{log_prefix}/hits5", hits5)
        self.log(f"{log_prefix}/hits3", hits3)
        self.log(f"{log_prefix}/hits1", hits1)
        self.log(f"{log_prefix}/mr", mr)
        self.log(f"{log_prefix}/mrr", mrr)

    def validation_epoch_end(self, outputs):
        self._eval_epoch_end(outputs, stage_name="Validation", log_prefix="Val")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_start(self):
        self.on_validation_start()

    def test_epoch_end(self, outputs):
        self._eval_epoch_end(outputs, stage_name="Test", log_prefix="Test")

    def configure_optimizers(self):
        fusion_params = (
            list(self.encoder.hv_fusion.parameters()) +
            list(self.encoder.gate_fc.parameters()) +
            list(self.timm.parameters()) +
            list(self.iimm.parameters())
        )
        fusion_ids = list(map(id, fusion_params))
        base_params = filter(lambda p: id(p) not in fusion_ids, self.parameters())
        optimizer_grouped_params = [
            {'params': base_params, 'lr': self.args.lr},
            {'params': fusion_params, 'lr': self.args.lr * 10}
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        return [optimizer], [scheduler]


LightningForMIMIC = LightningForEVEMEL
