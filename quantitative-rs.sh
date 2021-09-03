
cnt=0
for alt in {1..5}; do
    vals=()
    for dir in ./images/*; do
        cnt=$(($cnt + 1))

        val=$(find "$dir/case$alt/" | grep -e n2 -e target_origin.png | xargs psnr)
        vals+=($val)
    done
    metric=$(echo ${vals[*]} | xargs -n1 | average)
    echo "case $alt pnsr is $metric"
done

for alt in {1..5}; do
    vals=()
    for dir in ./images/*; do
        cnt=$(($cnt + 1))

        val=$(find "$dir/case$alt/" | grep -e n2 -e target_origin.png | xargs ssim)
        vals+=($val)
    done
    metric=$(echo ${vals[*]} | xargs -n1 | average)
    echo "case $alt ssmi is $metric"
done
