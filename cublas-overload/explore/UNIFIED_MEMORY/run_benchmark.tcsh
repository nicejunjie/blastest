#!/bin/tcsh -f

# a tcsh script to run assess CUDA's three memory access modes, namely conventional, zero-copy and unified. 
# blocksize: number of CUDA blocks
# threadsize: number of CUDA threads 
set i = 0
set blocksize = 1
set j = 0
set threadsize = 1

while ( $i <= 10 )

 set j = 0
 set threadsize = 1
 while ( $j <= 10 )
  
  foreach arraysize ( 1048576000 104857600 10485760 1048576 )
    echo "blocksize: " $blocksize
    echo "threadsize: " $threadsize 
    echo "arraysize: " $arraysize
    echo " "
    ./array_randomize_gpu $arraysize $blocksize $threadsize > output_"$arraysize"_"$blocksize"_"$threadsize"
  end

 @ threadsize = $threadsize * 2 
 @ j++
 end 
 
 @ blocksize = $blocksize * 2 
 @ i++
end 
