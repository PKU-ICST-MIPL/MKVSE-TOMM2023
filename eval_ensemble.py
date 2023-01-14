import logging
import argparse
import numpy as np
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)





def eval_ensemble(results_paths, fold5=False):
    all_sims = []
    all_npts = []
    for sim_path in results_paths:
        results = np.load(sim_path, allow_pickle=True).tolist()
        npts = results['npts']
        sims = results['sims']
        all_npts.append(npts)
        all_sims.append(sims)
    all_npts = np.array(all_npts)
    all_sims = np.array(all_sims)
    assert np.all(all_npts == all_npts[0])
    npts = int(all_npts[0])
    sims = all_sims.mean(axis=0)

    if not fold5:
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        npts = npts // 5
        results = []
        all_sims = sims.copy()
        for i in range(5):
            sims = all_sims[i * npts: (i + 1) * npts, i * npts * 5: (i + 1) * npts * 5]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])



def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities


def i2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False, mode='coco'):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/vse_infty/')
    parser.add_argument('--fold5', action='store_true',
                        help='MSCOCO 1K test settings.')
    parser.add_argument('--result1_path', default='/runs/MKVSE_coco_butd_region_bert_v3_1/results_coco.npy')
    parser.add_argument('--result2_path', default='/runs/MKVSE_coco_butd_region_bert_v12/results_coco.npy')
    opt = parser.parse_args()

    print(opt)

    # Evaluate model ensemble
    paths = [opt.result1_path, opt.result2_path]

    eval_ensemble(results_paths=paths, fold5=opt.fold5)


