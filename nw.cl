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
/*
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
*/
__kernel void
needleman_byblock(__global __read_only char* seq1,
                  __global __read_only char* seq2,
                  __global unsigned int *table,
                  __local int *buffer,
                  unsigned int iter,
                  int w, int h,
                  int buf_w, int buf_h,
                  int edge)
{
     // Global position of output pixel
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    // Load the relevant labels to a local buffer with a halo
    if ( x > 0 && y > 0 && x%(buf_w-edge) == 1 && y%(buf_h-edge) == 1 && x/(buf_w-edge) + y/(buf_h-edge) == iter ) {
        printf("iter:%u, x:%u, y:%u\n", iter, x, y);
        // load to local buffer
        for( int i=0; i<buf_w && x+i-edge<w; ++i ){
            buffer[to1D(buf_w, i, 0)] = table[to1D(w, x+i-edge, y-edge)];
        }
        for( int j=edge; j<buf_h && y+j-edge<h; ++j ){
            buffer[to1D(buf_w, 0, j)] = table[to1D(w, x-edge, y+j-edge)];
        }
        
        // do serial NW in local memory
        for( int j=edge; j<buf_h && y+j-edge<h; ++j ){
            for( int i=edge; i<buf_w && x+i-edge<w; ++i ){
                //printf("u:%u, l:%u\n", x-1+i-edge, y-1+j-edge);
                //printf("u:%c, l:%c\n", seq1[x-1+i-edge], seq2[y-1+j-edge]);
                //printf("take min, l:%u, u:%u\n", buffer[to1D(buf_w, i-1, j)], buffer[to1D(buf_w, i, j-1)] );
                buffer[to1D(buf_w, i, j)] = min( buffer[to1D(buf_w, i-1, j)]+1, buffer[to1D(buf_w, i, j-1)]+1 );
                //printf("temp min: %u\n", buffer[to1D(buf_w, i, j)]);
                int cur = buffer[to1D(buf_w, i-1, j-1)];
                //printf("lu: %u\n", cur);
                if( seq1[x-1+i-edge] != seq2[y-1+j-edge] ){
                    ++cur;
                }
                buffer[to1D(buf_w, i, j)] = min( buffer[to1D(buf_w, i, j)], cur );
                //printf("%u\n", buffer[to1D(buf_w, i, j)]);
            }
        }
        
        // copy local buffer back to global
        for( int j=1; j<buf_h && y+j-edge<h; ++j ){
            for( int i=1; i<buf_w && x+i-edge<w; ++i ){
                table[to1D(w, x+i-edge, y+j-edge)] = buffer[to1D(buf_w, i, j)];
            }
        }

    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    //barrier(CLK_LOCAL_MEM_FENCE);
}