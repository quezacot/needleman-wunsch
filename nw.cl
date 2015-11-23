inline int to1D(int w, int x, int y)
{
    return y*w + x;
}

inline int cuttail(int size, int v)
{
    if( v<0 )
        return 0;
    if( v >= size )
        return size-1;
    return v;
}

__kernel void
initialize_table(__global unsigned int *table,
                 int w, int h )
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    
    if( x<w && y<h ){
        if( x == 0 ){
            table[to1D(w,0,y)] = y;
        } else if( y == 0 ){
            table[to1D(w,x,0)] = x;
        } else{
            table[to1D(w,x,y)] = 0;
        }
    }
}

__kernel void
needleman_by1(__global __read_only char* seq1,
              __global __read_only char* seq2,
              __global unsigned int *table,
              unsigned int iter,
              int w, int h )
{
     // Global position of output pixel
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    // Load the relevant labels to a local buffer with a halo
    if ( x > 0 && y > 0 && x + y == iter ) {
        printf("iter:%u, x:%u, y:%u", iter, x, y);
        unsigned int pre = min( table[to1D(w, x-1, y)] + 1, table[to1D(w, x, y-1)] + 1 );
        unsigned int cur = table[to1D(w, x-1, y-1)];
        if( seq1[x-1] != seq2[y-1] ){
            ++cur;
        }
        table[to1D(w, x, y)] = min( pre, cur );
    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    //barrier(CLK_LOCAL_MEM_FENCE);
}
