cd ../..
th eval_box.lua \
    -dataset refcoco_licheng \
    -id 0 \
    -split val 2>&1 | tee ./job/refcoco_licheng/box_id0_val.log
