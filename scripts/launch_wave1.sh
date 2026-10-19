#!/bin/bash
# Launch Stage-1 wave-1 ablations in parallel, one per idle B200 node.
# sngpu --wrap mangles embedded args, so each job is a tiny per-run script.
set -e
cd /import/ml-sc-scratch5/chenw/MiniOneRec

launch () {
  local TAG=$1 NODE=$2; shift 2
  cat > /tmp/wave1_${TAG}.sh <<EOF
#!/bin/bash
bash /import/ml-sc-scratch5/chenw/MiniOneRec/scripts/train_eval_small.sh ${TAG} models/Qwen3-1.7B $*
EOF
  chmod +x /tmp/wave1_${TAG}.sh
  sngpu --jobname=sm_${TAG} --gputype=b200m180 --gpu 8 --cpu 32 --mem 400000 \
    --time 08:00:00 --nodelist=${NODE} \
    --output=/import/ml-sc-scratch5/chenw/MiniOneRec/logs/sm_${TAG}_%j.log \
    --filepath /tmp/wave1_${TAG}.sh | tail -1
}

launch recency  sc3-cp95 --use_recency True
launch profile  sc3-cp96 --use_profile_summary True
launch subcat   sc3-cp97 --use_subcategory True
launch neg3     sc3-cp98 --neg_ratio 3.0
launch hardneg8 sc3-cp100 --hard_neg_ratio 0.8
launch base     sc3-cp101 --use_chat_template False