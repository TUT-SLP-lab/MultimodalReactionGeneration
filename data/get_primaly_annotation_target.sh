target_dir=/home/mikawa/lab/MultimodalReactionGeneration2/data/annotations
ref_dir=/home/mikawa/lab/MultimodalReactionGeneration/data/unprocess/multimodal_dialogue
speaker="アビ|こん"
remove_speaker="にわとり"
allows="makikon|nanax|あおり|すずき|みどりsan"

rm primal_data.txt
ls $ref_dir | grep -E $speaker > primal_data_tmp.txt
ls -l $target_dir | grep '^d' | awk '{print $9}' > dir_list.txt
awk 'NR==FNR { dirs[$0]=1; next } { for (dir in dirs) if (index($0, dir)) print }' dir_list.txt primal_data_tmp.txt > primal_data_tmp2.txt
awk -v pat="$remove_speaker" '!($0 ~ pat)' primal_data_tmp2.txt > primal_data_tmp3.txt
awk -v allow="$allows" '($0 ~ allow)' primal_data_tmp3.txt > primal_data.txt
rm primal_data_tmp.txt
rm primal_data_tmp2.txt
rm primal_data_tmp3.txt
rm dir_list.txt