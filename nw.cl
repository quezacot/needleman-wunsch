// 2D index to 1D
inline int to1D(int w, int x, int y)
{
    return y*w + x;
}

// cut the outside index to the boundary
inline int cuttail(int size, int v)
{
    if( v<0 )
        return 0;
    if( v >= size )
        return size-1;
    return v;
}

// table initialization
__kernel void
initialize_table(__global unsigned int *table,
                 int w, int h )
{
    const unsigned int x = get_global_id(0)+1;
    const unsigned int y = get_global_id(1)+1;
    if( x<w && y<h ){
        if( x == 1 ){
            table[to1D(w,0,y)] = y;
        }
        if( y == 1 ){
            table[to1D(w,x,0)] = x;
        }
        if( x == 1 && y == 1 ){
            table[to1D(w,0,0)] = 0;
        }
    }
}

// implementation 1: no block parallel
__kernel void
needleman_by1(__global __read_only char* seq1,
              __global __read_only char* seq2,
              __global unsigned int *table,
              unsigned int iter,
              int w, int h )
{
    // Global position of DP table
    const unsigned int x = get_global_id(0)+1;
    const unsigned int y = get_global_id(1)+1;
    // initialize 0th row and 0th column
    if ( x < w && y < h && x + y == iter ) {
        //printf("iter:%u, x:%u, y:%u", iter, x, y);
        unsigned int pre = min( table[to1D(w, x-1, y)] + 1, table[to1D(w, x, y-1)] + 1 );
        unsigned int cur = table[to1D(w, x-1, y-1)];
        if( seq1[x-1] != seq2[y-1] ){
            ++cur;
        }
        table[to1D(w, x, y)] = min( pre, cur );
    }
}

// implementation 2: block-wise parallel, serial within blocks
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
     // Global position of DP table
    const unsigned int x = get_global_id(0)+edge;
    const unsigned int y = get_global_id(1)+edge;
    
    // Local position relative to (0, 0) in workgroup
    const unsigned int lx = get_local_id(0)+edge;
    const unsigned int ly = get_local_id(1)+edge;
    
    // Workgroup ID to (0, 0) in workgroup
    const unsigned int wx = get_group_id(0);
    const unsigned int wy = get_group_id(1);

    if ( x < w && y < h && lx == edge && ly == edge && wx + wy == iter ) {
        // printf("iter:%u, x:%u, y:%u\n", iter, x, y);
        // load to local buffer, use only workers that lx or ly == 0
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
                buffer[to1D(buf_w, i, j)] = min( buffer[to1D(buf_w, i-1, j)]+1, buffer[to1D(buf_w, i, j-1)]+1 );
                int cur = buffer[to1D(buf_w, i-1, j-1)];
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
    barrier(CLK_LOCAL_MEM_FENCE);
}

// implementation 3: block-wise parallel, parallel within blocks
__kernel void
needleman_byblockworker(__global __read_only char* seq1,
                        __global __read_only char* seq2,
                        __global unsigned int *table,
                        __local int *buffer,
                        unsigned int iter,
                        int w, int h,
                        int buf_w, int buf_h,
                        int edge)
{
     // Global position of DP table
    const unsigned int x = get_global_id(0)+edge;
    const unsigned int y = get_global_id(1)+edge;
    
    // Local position relative to (0, 0) in workgroup
    const unsigned int lx = get_local_id(0)+edge;
    const unsigned int ly = get_local_id(1)+edge;
    
    // Workgroup ID to (0, 0) in workgroup
    const unsigned int wx = get_group_id(0);
    const unsigned int wy = get_group_id(1);
    
    // initialize the 0th row and column in local buffer, identify the workers need to run in this iteration
    if( x < w && y < h && wx + wy == iter ){
        //printf("iter:%u, x:%u, y:%u\n", iter, x, y);
        //printf("iter:%u, wx:%u, wy:%u\n", iter, wx, wy);
        
        // load to local buffer
        if( lx == edge ){ // 0th column of local buffer
            buffer[to1D(buf_w, 0, ly)] = table[to1D(w, x-edge, y)];
        }
        if( ly == edge ){ // 0th row of local buffer
            buffer[to1D(buf_w, lx, 0)] = table[to1D(w, x, y-edge)];
        }
        if( lx == edge && ly == edge ){ // cell (0,0) of local buffer
            buffer[to1D(buf_w, 0, 0)] = table[to1D(w, x-edge, y-edge)];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
     
    // parallel NW, move forward one anti-diagonal each iteration
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
    
    // copy back to global buffer
    if( x < w && y < h && wx + wy == iter ){
        table[to1D(w, x, y)] = buffer[to1D(buf_w, lx, ly)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
