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

// table initialization for 1D row and column tables
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

// implementation 4: parallel with coalesced read/write
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
            buffer[to1D(buf_w, 0, ly)] = col_table[to1D(h, y, wx)];
        }
        if( ly == edge ){ // 0th row of local buffer
            buffer[to1D(buf_w, lx, 0)] = row_table[to1D(w, x, wy)];
        }
        if( lx == edge && ly == edge ){ // cell (0,0) of local buffer
            buffer[to1D(buf_w, 0, 0)] = row_table[to1D(w, x-edge, wy)];
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
        if( lx == buf_w-1 || x == w-1 ){ // copy back to column table
            col_table[to1D(h, y, wx+1)] = buffer[to1D(buf_w, lx, ly)]; // last column
            if( ly == edge ){ // first element in the last column
                col_table[to1D(h, y-1, wx+1)] = buffer[to1D(buf_w, lx, 0)];
            }
        }
        if( ly == buf_h-1 || y == h-1 ){ // copy back to row table
            row_table[to1D(w, x, wy+1)] = buffer[to1D(buf_w, lx, ly)]; // last row
            if( lx == edge ){ // first element in the last row
                row_table[to1D(w, x-1, wy+1)] = buffer[to1D(buf_w, 0, ly)];
            }
        }        
   }
   barrier(CLK_LOCAL_MEM_FENCE);
}
