import os
import argparse
from tools.functions import Storage


class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'autograph': self.__GRAPH
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        datasetParas = HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name]
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs[
            'unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                                 **dataArgs,
                                 **commonArgs,
                                 # **conv1d_embedding,
                                 # **xlstm,
                                 **datasetParas,
                                 # **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                                 ))

    def __datasetCommonParams(self):
        root_dataset_dir = '/data/student_code/XHY/dataset/'
        tmp = {
            'mosi': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    't_size_in': 768,
                    'v_size_in': 20,
                    'a_size_in': 5,
                    'feature_dims': (768, 20, 5),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'epochs': 100
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    't_size_in': 768,
                    'v_size_in': 20,
                    'a_size_in': 5,
                    'feature_dims': (768, 20, 5),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'epochs': 70
                }
            },
            'mosei': {
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    't_size_in': 768,
                    'v_size_in': 35,
                    'a_size_in': 74,
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'epochs': 60
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    't_size_in': 768,
                    'v_size_in': 35,
                    'a_size_in': 74,
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss',
                    'epochs': 30
                }
            },
            'sims': {
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'CH-SIMSV2/CH-SIMSV2(S)/unaligned.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55),  # (text, audio, video)
                    't_size_in': 768,
                    'v_size_in': 177,
                    'a_size_in': 25,
                    'feature_dims': (768, 177, 25),  # (text, audio, video)
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                    'epochs': 60
                }
            }
        }
        return tmp

    def __GRAPH(self):
        tmp = {
            'commonParas': {
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_robert': False,
                'use_finetune': True,
                'save_labels': False,
                'output_attention': True,
                'output_hidden_state': False,
                'attn_mask': False,
                # 'early_stop': 8,
                # 'update_epochs': 4
            },
            'datasetParas': {
                'mosi': {
                    't_in_dim':128,
                    'v_in_dim':128,
                    'a_in_dim':128,
                    't_emb_depth':1,
                    'v_emb_depth':1,
                    'a_emb_depth':1,
                    't_emb_heads':12,
                    'v_emb_heads':12,
                    'a_emb_heads':12,
                    't_emb_dim':64,
                    'v_emb_dim':64,
                    'a_emb_dim':64,
                    't_mlp_dim':512,
                    'v_mlp_dim':512,
                    'a_mlp_dim':512,
                    'gt_in_dim': 256,
                    'gt_du_in_dim': 256 * 2,
                    'gt_dim': 256,
                    'gt_out_dim': 256,
                    'gt_heads': 4,
                    'gt_topk': 32,
                    'attention_dropout_prob': 0.2,

                    'grab_in_dim': 256,
                    'grab_heads': 4,
                    'grab_att_dim': 256,
                    'grab_out_dim': 256,
                    'grab_dropout_prob': 0.2,
                    'grab_inter_dim': 768,
                    'grab_layers': 4,

                    # single modal
                    'projs1_t_t': 786,
                    'projs2_t_t': 256,
                    'projs_out_t_t': 1,
                    'projs1_v_v': 786,
                    'projs2_v_v': 256,
                    'projs_out_v_v': 1,
                    'projs1_a_a': 786,
                    'projs2_a_a': 256,
                    'projs_out_a_a': 1,
                    'projs1_s': 768,
                    'projs2_s': 256,
                    'proj_out_s': 1,
                    # multimodal
                    'projc1_t_va': 786,
                    'projc2_t_va': 256,
                    'projc_out_t_va': 1,
                    'projc1_v_ta': 786,
                    'projc2_v_ta': 256,
                    'projc_out_v_ta': 1,
                    'projc1_a_tv': 786,
                    'projc2_a_tv': 256,
                    'projc_out_a_tv': 1,
                    'projc1_f': 786,
                    'projc2_f': 256,
                    'projc_out_f': 1,
                    'output_dropout': 0.2,
                    # 'uncertainly':{
                    'hidden_sz': 256,
                    # version
                    'V_dim':[128, 256, 512],
                    'V_out_dim':[256, 512, 768],
                    'V_depth':2,
                    'V_heads':[4, 8, 12],
                    'V_mlp_dim':512,
                    'dis_depth':2,
                    'dis_in_feature':[512, 256, 128, 64],
                    'dis_hidden_size':512,
                    # recall
                    'rc_dim':[512, 256, 128],
                    'rc_out_dim':[256, 128,],
                    'rc_depth':2,
                    'rc_heads':[8, 4, 2],
                    'rc_mlp_dim':512,

                },

                'mosei': {
                    't_in_dim':128,
                    'v_in_dim':128,
                    'a_in_dim':128,
                    't_emb_depth':1,
                    'v_emb_depth':1,
                    'a_emb_depth':1,
                    't_emb_heads':12,
                    'v_emb_heads':12,
                    'a_emb_heads':12,
                    't_emb_dim':64,
                    'v_emb_dim':64,
                    'a_emb_dim':64,
                    't_mlp_dim':512,
                    'v_mlp_dim':512,
                    'a_mlp_dim':512,
                    'gt_in_dim': 256,
                    'gt_du_in_dim': 256 * 2,
                    'gt_dim': 256,
                    'gt_out_dim': 256,
                    'gt_heads': 4,
                    'gt_topk': 32,
                    'attention_dropout_prob': 0.2,

                    'grab_in_dim': 256,
                    'grab_heads': 4,
                    'grab_att_dim': 256,
                    'grab_out_dim': 256,
                    'grab_dropout_prob': 0.2,
                    'grab_inter_dim': 768,
                    'grab_layers': 4,

                    # single modal
                    'projs1_t_t': 786,
                    'projs2_t_t': 256,
                    'projs_out_t_t': 1,
                    'projs1_v_v': 786,
                    'projs2_v_v': 256,
                    'projs_out_v_v': 1,
                    'projs1_a_a': 786,
                    'projs2_a_a': 256,
                    'projs_out_a_a': 1,
                    'projs1_s': 768,
                    'projs2_s': 256,
                    'proj_out_s': 1,
                    # multimodal
                    'projc1_t_va': 786,
                    'projc2_t_va': 256,
                    'projc_out_t_va': 1,
                    'projc1_v_ta': 786,
                    'projc2_v_ta': 256,
                    'projc_out_v_ta': 1,
                    'projc1_a_tv': 786,
                    'projc2_a_tv': 256,
                    'projc_out_a_tv': 1,
                    'projc1_f': 786,
                    'projc2_f': 256,
                    'projc_out_f': 1,
                    'output_dropout': 0.2,
                    # 'uncertainly':{
                    'hidden_sz': 256,
                    # version
                    'V_dim':[128, 256, 512],
                    'V_out_dim':[256, 512, 768],
                    'V_depth':2,
                    'V_heads':[4, 8, 12],
                    'V_mlp_dim':512,
                    'dis_depth':1,
                    'dis_in_feature':[512, 256, 128],
                    'dis_hidden_size':512,
                    # recall
                    'rc_dim':[512, 256, 128],
                    'rc_out_dim':[256, 128,],
                    'rc_depth':2,
                    'rc_heads':[8, 4, 2],
                    'rc_mlp_dim':512,
                },
                'sims': {
                    'gt_in_dim': 256,
                    'gt_du_in_dim': 256 * 2,
                    'gt_dim': 256,
                    'gt_out_dim': 256,
                    'gt_heads': 4,
                    'gt_topk': 16,
                    'attention_dropout_prob': 0.5,

                    'grab_in_dim': 256,
                    'grab_heads': 4,
                    'grab_att_dim': 256,
                    'grab_out_dim': 256,
                    'grab_dropout_prob': 0.5,
                    'grab_inter_dim': 768,
                    'grab_layers': 2,

                    # single modal
                    'projs1_t_t': 786,
                    'projs2_t_t': 256,
                    'projs_out_t_t': 1,
                    'projs1_v_v': 786,
                    'projs2_v_v': 256,
                    'projs_out_v_v': 1,
                    'projs1_a_a': 786,
                    'projs2_a_a': 256,
                    'projs_out_a_a': 1,
                    'projs1_s': 768,
                    'projs2_s': 256,
                    'proj_out_s': 1,
                    # multimodal
                    'projc1_t_va': 786,
                    'projc2_t_va': 256,
                    'projc_out_t_va': 1,
                    'projc1_v_ta': 786,
                    'projc2_v_ta': 256,
                    'projc_out_v_ta': 1,
                    'projc1_a_tv': 786,
                    'projc2_a_tv': 256,
                    'projc_out_a_tv': 1,
                    'projc1_f': 786,
                    'projc2_f': 256,
                    'projc_out_f': 1,
                    'output_dropout': 0.5,
                    # 'uncertainly':{
                    'hidden_sz': 256,

                },
            },
        }

        return tmp

    def get_config(self):
        return self.args