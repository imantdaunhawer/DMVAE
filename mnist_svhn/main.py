import glob
import uuid
import re
import json
import os
import shutil
import argparse
import torchvision
import numpy as np
from training import training_procedure
import resource
from utils import LIKELIHOOD_DICT
from getters import Getters

parser = argparse.ArgumentParser()

# meta arguments
parser.add_argument('--data_dir', type=str, default="./mnist_svhn/data/MNIST_SVHN", help="path to datadir")
parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
parser.add_argument('--debug', default=False, action="store_true", help="run in debug mode")
parser.add_argument('--random_seed', type=int, default=None, help="random seed")
parser.add_argument('--num_modalities', type=int, default=None, help="number of modalities")
parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loaders")
parser.add_argument('--num_imp_samples', type=int, default=5000, help="number of importance samples for evaluation")
parser.add_argument('--likelihood_str', type=str, help="likelihoods to be used in the decoders, e.g. 'bernoulli-normal-laplace'", required=True)
parser.add_argument('--reparam_c_for_eval', default=False, action="store_true", help="reparameterize content representation (for evaluation only)")

# optimization arguments
parser.add_argument('--end_epoch', type=int, help="number of training epochs")
parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")

# TC discriminator arguments
parser.add_argument('--tc_hidden_size', type=int, default=1000, help="TC discriminator hidden layer size")
parser.add_argument('--tc_initial_lr', type=float, default=0.0001, help="TC discriminator starting learning rate")
parser.add_argument('--tc_beta_1', type=float, default=0.5, help="default beta_1 val for adam for TC discriminator")
parser.add_argument('--tc_beta_2', type=float, default=0.9, help="default beta_2 val for adam for TC discriminator")

# dimensionality arguments
parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=10, help="dimension of common factor latent space")

# loss coefficients
parser.add_argument('--reconstruction_coefs', type=float, nargs="+", help="coefficients for reconstruction terms")
parser.add_argument('--beta_prior_styles', type=float, nargs="+", help="modality-specific coefficients for KL-Divergence to prior")
parser.add_argument('--beta_prior_content', type=float, help="shared coefficient for KL-Divergence to prior")
parser.add_argument('--beta_condreg', type=float, help="coefficient for KL-Divergence between posteriors (conditional likelihood regularizer)")
parser.add_argument('--infomax_coef', type=float, default=0., help="coefficient for infomax loss term")
parser.add_argument('--disentanglement_coefs', type=float, nargs="+", default=0., help="coefficient for disentanglement loss term for each modality")
parser.add_argument('--llik_scale_param', type=float, nargs="+", help="likelihood scale parameter")

# annealing coefficients
parser.add_argument('--annealing_epochs', type=float, default=None,
                    help="divergence annealing: number of epochs to reach max weight")
parser.add_argument('--start_annealing', type=float, default=0.,
                    help="divergence annealing: first epoch when annealing starts")
parser.add_argument('--anneal_c_only', default=False, action="store_true",
                    help="If true, anneal only the KLD for content, otherwise anneal it for content and style.")

# meta-training flags
parser.add_argument('--prior_expert', default=False, action="store_true", help="whether to use a prior expert in the POE")
parser.add_argument('--cm_dropout', default=False, action="store_true", help="whether to use c_m-dropout")
parser.add_argument('--noisy_inputs', default=False, action="store_true", help="whether to augment inputs with Gaussian white noise")
parser.add_argument('--disentangle_style_only', default=False, action="store_true",
    help="whether to backprop the disentanglement loss through style only. If false, backprop through style and content.")
parser.add_argument('--disjoint_partition', default=False, action="store_true",
    help="whether to use a disjoint partition instead of the full multimodal posterior")
parser.add_argument('--freeze_content', default=False, action="store_true", help="freeze content representation before its fed to decoders")
parser.add_argument('--reparam_c_for_decoders', default=False, action="store_true", help="reparameterize content representation (for decoders during training)")
parser.add_argument('--infomax_nonlinear_projection_head', default=False, action="store_true", help="use nonlinear projection head, instead of a default linear one")
parser.add_argument('--contrast_full_vs_subset', default=False, action="store_true", help="For contrasting, use full-set vs. random subset")
parser.add_argument('--reparam_c_before_infomax', default=False, action="store_true", help="For contrasting, use reparameterized representations")

# logging arguments
parser.add_argument('--load_saved', default=False, action="store_true",
                    help="flag to indicate if a saved model will be loaded")
parser.add_argument('--saved_path', type=str, default='checkpoints', help="path to saved model")
parser.add_argument('--encoder_file', type=str, default='encoder', help="filename for encoder")
parser.add_argument('--decoder_file', type=str, default='decoder', help="filename for decoder")
parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")
parser.add_argument('--log_dir', type=str, required=True, help="directory to save tensorboard logs")
parser.add_argument('--eval_freq_likelihood', type=int, default=np.inf,
                    help="evaluation frequency for likelihoods (every n-th epoch). Deactivated by default.")
parser.add_argument('--eval_freq_generation', type=int, default=np.inf,
                    help="evaluation frequency for generation (every n-th epoch). Deactivated by default.")
parser.add_argument('--save_freq', type=int, default=10, help="save frequency: save models every n-th epoch")
parser.add_argument('--num_prior_samples', type=int, default=0, help="number of prior samples for swapping plots")
parser.add_argument('--log-dir-versioning', default=False, action="store_true",
                    help="flag to indicate if tensorboard logs should be versioned")
parser.add_argument('--eval_freq_fid', type=int, default=np.inf,
                    help="evaluation frequency for fid-scores (every n-th epoch). Deactivated by default.")
parser.add_argument('--num_imgs_fid', type=int, default=10000, help="number of images to perform fid evaluation on")
parser.add_argument('--log_classification', default=False, action="store_true", help="log classification performance")

FLAGS = parser.parse_args()


if __name__ == '__main__':

    # fixes pytorch memory bug (0 items of ancdata)
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # fixes sporadic plotting problems
    # https://github.com/ipython/ipython/issues/10627
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    # check likelihoods
    likelihoods = []
    for l in FLAGS.likelihood_str.split("-"):
        likelihoods.append(LIKELIHOOD_DICT[l])
    if len(likelihoods) == 1:  # assume similar likelihoods if only one was provided
        tmp = likelihoods[0]
        likelihoods = [tmp for _ in range(FLAGS.num_modalities)]
    assert len(likelihoods) == FLAGS.num_modalities

    # for parameters that depend on the number of modalities, if there is a single value instead of a list,
    # broadcast that value to all modalities (e.g., assume equal reconstruction coefs)
    for attr in ["reconstruction_coefs", "llik_scale_param", "beta_prior_styles", "disentanglement_coefs"]:
        values = getattr(FLAGS, attr)
        if len(values) == 1:
            new_values = [values[0] for _ in range(FLAGS.num_modalities)]  # broadcast single value
            setattr(FLAGS, attr, new_values)

    # set up logdir
    if FLAGS.log_dir_versioning:
        version_dirs = glob.glob("%s/version_*" % FLAGS.log_dir)
        r = re.compile(r"\d+")
        versions = [int(r.findall(d)[-1]) for d in version_dirs]
        versions = sorted(versions)
        new_version = versions[-1] + 1 if len(versions) > 0 else 1
        new_logdir = "%s/version_%d" % (FLAGS.log_dir, new_version)
        FLAGS.log_dir = new_logdir
    # w/o versioning, if logdir already exists, remove it
    else:
        try:
            print("[WARN] removing tensorboard logdir '%s'" % FLAGS.log_dir)
            shutil.rmtree(FLAGS.log_dir)
        except FileNotFoundError:
            pass
    os.makedirs(FLAGS.log_dir)  # TODO: catch FileExistsError
    print("LOGDIR:", FLAGS.log_dir)

    # save flags to the log_dir
    with open("%s/flags.json" % FLAGS.log_dir, "w") as fp:
        json.dump(vars(FLAGS), fp)

    # create paths for (un-)conditional generation
    if FLAGS.eval_freq_fid > 0:
        gtrs = Getters()
        train, test = gtrs.get_data_loaders(batch_size=FLAGS.batch_size, num_modalities=FLAGS.num_modalities,
           num_workers=FLAGS.num_workers, shuffle=True, device="cuda", data_dir=FLAGS.data_dir)
        hash_value = str(uuid.uuid4())
        gen_path = "/tmp/%s/generated_images" % hash_value
        test_paths = ["/tmp/%s/test_images/m%d" % (hash_value, m) for m in range(FLAGS.num_modalities)]
        for p in [gen_path, *test_paths]:
            if not os.path.exists(p):
                os.makedirs(p)
                print("Created new directory:", p)
        # save test data for each modality
        for m, p in enumerate(test_paths):
            if list(glob.glob("%s/precomputed_fid_statistics.npz" % p)):
                print("Found pre-computed FID-statistics")
                continue
            else:
                print("For FID computation, save test images for m=%d to %s" % (m, p))
                cnt = 0
                for batch in test:
                    for i in range(FLAGS.batch_size):
                        image = batch[m][0][i, :, :, :]
                        label = batch[m][1][i]
                        torchvision.utils.save_image(image, "{}/{}_{}.png".format(p, cnt, int(label)))
                        cnt += 1
                    if cnt > FLAGS.num_imgs_fid:
                        break  # NOTE: take at most x test images (e.g., if the evaluation of FID scores takes too long)
        FLAGS.fid_gen_path = gen_path
        FLAGS.fid_test_paths = test_paths

    # copy source files for reproducibility
    shutil.copy2("../main.py", FLAGS.log_dir)
    shutil.copy2("../utils.py", FLAGS.log_dir)
    shutil.copy2("../training.py", FLAGS.log_dir)
    shutil.copy2("networks.py", FLAGS.log_dir)
    shutil.copy2("getters.py", FLAGS.log_dir)

    # run training
    training_procedure(FLAGS)
