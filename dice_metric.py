import boxx
from boxx import np
from pycocotools.coco import COCO
import pycocotools.cocoeval


class COCOevalWithObjectDice(pycocotools.cocoeval.COCOeval):
    def summarize(self):
        super().summarize()
        scocoDic, simgdf, sanndf, scatdf = boxx.loadCoco(self.cocoDt.dataset)
        gcocoDic, gimgdf, ganndf, gcatdf = boxx.loadCoco(self.cocoGt.dataset)

        S = boxx.defaultdict(lambda: {})
        G = boxx.defaultdict(lambda: {})
        for imgid in simgdf["id"]:
            sanns = boxx.df2dicts(sanndf[sanndf.image_id == imgid])
            smasks = np.array(list(map(self.cocoDt.annToMask, sanns)))
            tanns = boxx.df2dicts(ganndf[ganndf.image_id == imgid])
            gmasks = np.array(list(map(self.cocoGt.annToMask, tanns)))

            # smasks = smasks[:-2]
            # gmasks = gmasks[3:]

            s_area = smasks.sum(-1).sum(-1)
            g_area = gmasks.sum(-1).sum(-1)

            s_g_to_overlap = (smasks[:, None] * gmasks[None]).sum(-1).sum(-1)
            g_index_of_best_match = s_g_to_overlap.argmax(1)

            s_tp = s_g_to_overlap[range(len(smasks)), g_index_of_best_match]
            s_fp = s_area - s_tp
            s_fn = g_area[g_index_of_best_match] - s_tp
            s_dice = 2 * s_tp / (2 * s_tp + s_fp + s_fn)
            [
                S[(imgid, i)].update(dice=s_dice[i], area=s_area[i])
                for i in range(len(smasks))
            ]

            s_index_of_best_match = s_g_to_overlap.argmax(0)
            g_tp = s_g_to_overlap[s_index_of_best_match, range(len(gmasks))]
            g_fn = g_area - g_tp
            g_fp = s_area[s_index_of_best_match] - g_tp
            g_dice = 2 * g_tp / (2 * g_tp + g_fp + g_fn)
            [
                G[(imgid, i)].update(dice=g_dice[i], area=g_area[i])
                for i in range(len(gmasks))
            ]

        dice_S = sum([d["dice"] * d["area"] for d in S.values()]) / sum(
            [d["area"] for d in S.values()]
        )

        dice_G = sum([d["dice"] * d["area"] for d in G.values()]) / sum(
            [d["area"] for d in G.values()]
        )

        re = dict(object_dice=(dice_S + dice_G) / 2)
        print(re)
        boxx.mg()
        return re


class COCOevalWithSemanticDice(pycocotools.cocoeval.COCOeval):
    def summarize(self):
        super().summarize()
        cocoDic, imgdf, anndf, catdf = boxx.loadCoco(self.cocoDt.dataset)
        tcocoDic, timgdf, tanndf, tcatdf = boxx.loadCoco(self.cocoGt.dataset)
        tp_ = 0
        fp_ = 0
        fn_ = 0
        dices_ = []
        for imgid in imgdf["id"]:
            anns = boxx.df2dicts(anndf[anndf.image_id == imgid])
            mask = sum(map(self.cocoDt.annToMask, anns)) > 0
            tanns = boxx.df2dicts(tanndf[tanndf.image_id == imgid])
            tmask = sum(map(self.cocoGt.annToMask, tanns)) > 0
            # if imgid%2:tmask[:30] = 1
            # if imgid%4:mask[:, :30] = 1
            tp = (mask * tmask).sum()
            fp = mask.sum() - tp
            fn = tmask.sum() - tp
            tp_ += tp
            fp_ += fp
            fn_ += fn
            dices_.append(2 * tp / (2 * tp + fp + fn))

        re = dict(
            all_pixel_dice=2 * tp_ / (2 * tp_ + fp_ + fn_), mean_dice=np.mean(dices_)
        )
        print(re)
        boxx.mg()
        return re


if __name__ == "__main__":
    from boxx import *

    # pr_coco_file, gt_coco_file = (
    #     "test_example/aug_val_del/gland_2test-dev_results.segm.json",
    #     "test_example/aug_val_del/instances_val2017.json",
    # )

    pr_coco_file, gt_coco_file = (
        "test_example/gland_noaugtest-dev_results.segm.json",
        "test_example/instances_val2017.json",
    )
    # pr_coco_file, gt_coco_file = "test_example/by_iseg/pr.json", "test_example/by_iseg/gt.json"
    cocoGt = COCO(gt_coco_file)
    cocoDt = cocoGt.loadRes(pr_coco_file)

    evalmethod = COCOevalWithObjectDice(cocoGt, cocoDt, "segm")
    evalmethod.evaluate()
    evalmethod.accumulate()
    re = evalmethod.summarize()
