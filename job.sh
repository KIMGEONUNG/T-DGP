#!/bin/bash

for f in $(find ./datasets/img_t -name '*.jpg' | sort -R); do

    id=${f#*img_t/[0-9][0-9]/}
    id=${id%.jpg}
    
    class=${f#*img_t/}
    class=${class%/*.jpg}
    echo "target file is $f"
    echo "id is $id"
    echo "class is $class"

    for case_i in {1..5}; do

        target_dir="./images/${id}_${class}/case${case_i}"
        echo target directory is $target_dir
        mkdir -vp $target_dir

        ./autogen.sh $f $class $case_i

        mv -v ./images/*.png $target_dir
    done
done
