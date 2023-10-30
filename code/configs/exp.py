from easydict import EasyDict as edict

config = edict()
config.network = "swin_t"
config.resume = False
config.resume_step = 0
config.init = True
config.init_model = "./"  # TODO
config.save_all_states = True
config.output = "./results"  # TODO

config.img_size = 112
config.embedding_size = 512
config.fp16 = True

# For SGD
#config.optimizer = "sgd"
#config.lr = 0.1
#config.momentum = 0.9
#config.weight_decay = 5e-4

# For AdamW
config.optimizer = "adamw"
config.lr = 5e-4
config.weight_decay = 0.05

config.lr_name = 'cosine'
config.warmup_lr = 5e-7
config.min_lr = 5e-6

# Epoch interval to decay LR, used in StepLRScheduler
config.decay_epoch = 10
# LR decay rate, used in StepLRScheduler
config.decay_rate = 0.1

config.verbose = 1000
config.save_verbose = 6000
config.frequent = 10


# For Large Sacle Dataset, such as WebFace42M
config.dali = False

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 0

config.warmup_step = 2000
####Raf-db
# config.total_step = 20000
####Affectnet
config.total_step = 60000

# Data

config.expression_train_dataset = "RAF-DB"
config.expression_val_dataset = "RAF-DB"
# config.expression_train_dataset = "AffectNet"
# config.expression_val_dataset = "AffectNet"


config.RAF_data = "dataset/RAF"
config.RAF_label = "dataset/list_patition_label.txt"



config.standard_train_sample_num = 1000000

config.INTERPOLATION = 'bicubic'

config.RAF_NUM_CLASSES = 7
# Label Smoothing
config.RAF_LABEL_SMOOTHING = 0.1


config.AUG_COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
config.AUG_AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# config.AUG_AUTO_AUGMENT = 'none'
# Random erase prob
#0.25
config.AUG_REPROB = 0.25
# Random erase mode
config.AUG_REMODE = 'pixel'
# Random erase count
config.AUG_RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
config.AUG_MIXUP = 0.0 #0.8
# Cutmix alpha, cutmix enabled if > 0
config.AUG_CUTMIX = 0.0 #1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
config.AUG_CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
config.AUG_MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
config.AUG_MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
config.AUG_MIXUP_MODE = 'batch'

config.AUG_SCALE_SET = True
config.AUG_SCALE_SCALE = (1.0, 1.0)
config.AUG_SCALE_RATIO = (1.0, 1.0)


# config.batch_size = 64
config.batch_size = 32
config.train_num_workers = 2
config.train_pin_memory = True

config.val_batch_size = 64
config.val_num_workers = 0
config.val_pin_memory = True


