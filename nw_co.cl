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
initialize_table(__global unsigned int *row_table,
                 __global unsigned int *col_table,
                 int w, int h )
{
    const unsigned int x = get_global_id(0)+1;
    const unsigned int y = get_global_id(1)+1;
    if( x<w && y<h ){
        if( x == 1 ){
            col_table[to1D(h,y,0)] = y;
        }
        if( y == 1 ){
            row_table[to1D(w,x,0)] = x;
        }
        if( x == 1 && y == 1 ){
            col_table[to1D(h,0,0)] = 0;
            row_table[to1D(w,0,0)] = 0;
        }
    }
}

__kernel void
needleman_coalesce(__global __read_only char* seq1,
                   __global __read_only char* seq2,
                   __global unsigned int *row_table,
                   __global unsigned int *col_table,
                   __local int *buffer,
                   unsigned int iter,
                   int w, int h,
                   int buf_w, int buf_h,
                   int edge )
{
     // Global position of output pixel
    const unsigned int x = get_global_id(0)+edge;
    const unsigned int y = get_global_id(1)+edge;
    
    // Local position relative to (0, 0) in workgroup
    const unsigned int lx = get_local_id(0)+edge;
    const unsigned int ly = get_local_id(1)+edge;
    
    // Workgroup ID to (0, 0) in workgroup
    const unsigned int wx = get_group_id(0);
    const unsigned int wy = get_group_id(1);
    
    // Load the relevant labels to a local buffer with a halo
    if( x < w && y < h && wx + wy == iter ){
        //printf("iter:%u, x:%u, y:%u\n", iter, x, y);
        //printf("iter:%u, wx:%u, wy:%u\n", iter, wx, wy);
        // load to local buffer
        
        if( lx == edge ){
            buffer[to1D(buf_w, 0, ly)] = col_table[to1D(h, y, wx)];
        }
        if( ly == edge ){
            buffer[to1D(buf_w, lx, 0)] = row_table[to1D(w, x, wy)];
        }
        if( lx == edge && ly == edge ){
            buffer[to1D(buf_w, 0, 0)] = row_table[to1D(w, x-edge, wy)];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
        
    for( int i=2; i < buf_w + buf_h - 1; ++i ){
        if( x < w && y < h && wx + wy == iter && lx + ly == i ){
            //printf("fill %d: iter:%u, lx:%u, ly:%u\n", i, iter, lx, ly);
            int cur;
            if( seq1[x-1] == seq2[y-1] ){
                cur = buffer[to1D(buf_w, lx-1, ly-1)];
            } else{
                cur = buffer[to1D(buf_w, lx-1, ly-1)] + 1;
            }
            buffer[to1D(buf_w, lx, ly)] = min( buffer[to1D(buf_w, lx-1, ly)]+1, buffer[to1D(buf_w, lx, ly-1)]+1 );
            buffer[to1D(buf_w, lx, ly)] = min( buffer[to1D(buf_w, lx, ly)], cur );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if( x < w && y < h && wx + wy == iter ){
        if( lx == buf_w-1 || x == w-1 ){
            col_table[to1D(h, y, wx+1)] = buffer[to1D(buf_w, lx, ly)];
            if( ly == edge ){
                col_table[to1D(h, y-1, wx+1)] = buffer[to1D(buf_w, lx, 0)];
            }
        }
        if( ly == buf_h-1 || y == h-1 ){
            row_table[to1D(w, x, wy+1)] = buffer[to1D(buf_w, lx, ly)];
            if( lx == edge ){
                row_table[to1D(w, x-1, wy+1)] = buffer[to1D(buf_w, 0, ly)];
            }
        }        
   }
   barrier(CLK_LOCAL_MEM_FENCE);
}
