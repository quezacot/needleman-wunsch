from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import math
import pylab
import string
import time

# Edit distance = 1 for insertion, deletion, and replacement
# Serial version
def edit_distance(strA, strB):
    lenA, lenB = len(strA), len(strB)
    if strA and strB:
        # DP table
        # DP[i,j] = edit distance between strA[:(i+1)] and strB[:(j+1)]
        dp = np.zeros((lenA+1, lenB+1), dtype=np.uint)
        # Initialize 
        dp[:,0] = np.arange(lenA+1, dtype=np.uint)
        dp[0,:] = np.arange(lenB+1, dtype=np.uint)
        # Update
        for i in range(1, lenA+1):
            for j in range(1, lenB+1):
                dp[i,j] = min(dp[i-1,j]+1,
                               dp[i,j-1]+1,
                               dp[i-1,j-1] + (0 if strA[i-1] == strB[j-1] else 1))
        return dp.T
        #return dp[lenA, lenB]
    else:
        return max(lenA, lenB)

# round up the global size to make sure it is dividable
def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r
        
        
if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    #context = cl.Context(devices[2:])
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('nw_co.cl').read()).build(options='')
    # different size of strings to test DNA sequences
    # string1 = "CAGCATATACGCGCTCGCCCGCCGCCGCTGGCTCGCTAAGGGTACGCTTGCCTGAGCCTC \
                # TGCCAATGTCGCCTGTTCATTGGGAAGGCGGAGGTAACGTGCGCAGTCCGCGTAGCTGCC \
                # AGAGTCTTCATGGAGGAGGTTGAGCGCAGCGGGCCACAAGTACCCGGGGCGGCCCGGGGG \
                # GGGCGGAGGACGTCGGGGCAGCAACCGCAATCGGCCGTCGACCGACCTCGGACACAAGGT \
                # TGATCGTGACGCCCAGGCCGAGCCGGAAGTGGGCGAGTGTGACTTCGAACTACTACGCTT \
                # CAGCACTTGCCCGTAGACCCAGGCACCGCCCCCCTGGTGGGTTGACGCCGCGTGCGCTAG \
                # TTCCGGGCCCACCAACCGGTATGATGGGAGACCGGCATTGCAGCACGGCCTCCCGCGACG \
                # TAGACGCCGCCCCAACCATCCGGCAGCTCTTGCACGCGCGGGTGCAGGATCGGGTCGGAT \
                # GTGAGGAGAGGGGACTGCCTCGTCAGTGCCTCAAACTCCAAGGCAGACGGAAAACCCTGA \
                # CGGTTCTCAGTAATGGAAATGGCCAAGGCAGCCCGCAGAACGAGGAGATCCTGGCACCAC \
                # CCGCACTTTGCTAAAGGCCCGTAACTGTACGGCCTACACCGTGACTACGGCCGTAGGGGC \
                # CCCCCCACAGATGTACGTACCGACCAAGGGCAGACGGGTCGGCGGCAGCGGACTGCACGC \
                # CGCGACCTTAGTCCCCCTCCGCGGCGCCGCATAGACCGGAAACCGGCGTCTCCAGAGACA \
                # CCGGCGCTGCCTGATTGGCGTGGGAAAATTCGGCTCGGGCCGCGTAGCGAGCGTAGCCTG \
                # GCGCGCGCGGGAGCCACTGTGGTTGCAGTAAGCGCCGTGGACGTCGGCGATGTTCGCGGA \
                # ATAGTTGCGGCCATTCGGCTGACCGTGGACTGGTGTAAAGAATCGGGCACAGAAACGGAG \
                # CAGGACGGGGAGGAAACAATCTGGAATGCGGTCTCTGGCC"
    # string2 = "CCTAGGATACGCGCTAACCAGACGTTTGGGGAGACGTGGCGAAGTGAGGCGCTGCCGCCG \
                # CTAAAGCGCCTCCCGGGCTCTCCCCCTCGAAGAACGATTGTGCGTAATGCAAGGGCTCTA \
                # TAGGCAAGATGCGGAACATGATGAGCAGCAGGCGCTACGGCTTGCACAGAACGGCCCCGT \
                # AAATCCCCTATGGGTCCTTTGTAAGGTTAGGAATGGCCTCACCTGACTGCAGTAACGTCC \
                # ATCACGTGGCGGGATACACCCCCAGCCCGCTCGCGTGAGCTCCGGTGCCACTCAGACGGT \
                # CGCCGGCCGCGGTCATCGTTGGGCCTCTTAATGTGGGTACAAGGGTCAGTTCTGGACCGG \
                # GGTCACAATTTTGGGGTCGGCCACGAAGGTTTCGATTCCCTGACGTCGCGGGGCGCCTGT \
                # GACCCCAGCGACGATGGCGGCGGTCCGACGGTCGAATGAGCGCATCGACCGGCGACTGGT \
                # ATTAAGGTCCCGGGGGGCCTGCAGGCATATACGCCGGCCAGACCGCGTAACAACGTCAGG \
                # GTTGCCCTTGCGGCACTCCGGGGGTCGCCTTCCCCGACACTCGGGCAGCGGTGCACTCCT \
                # ACTGGCTTGTTCGAACATGCTGCCAGAAGCCTCTCCGCCCGTAGTCATCCCCGACGCGTC \
                # TGATTGCCTCAGCCCGTTGTTCGCATGGGCATCGCTTAGCGTCGCCAGACGGAGGATCCG \
                # GGACGGCAAAGGGAAATGCCGTGAGTGCGCGCGCGCCCTCCGGCCGCCCAGGGGAGACCA \
                # CGCTCTTTCCATCGTAACGGGCCGGGCCGCTCCTGCTGCTTACCGGAACGCATCCCTGAG \
                # CGCGGAACGCTACATAGGCGGGACGCCGCAGAGCGCGGCTTCCACGACGGGGCGCGAACC \
                # ATTTTGCATAGGATTGAGAGCCCGGCGAGCCCTTGAGCACGACCCCCCATGGACCCATCG \
                # TCTGGTCAGGTGGGCAAACCGACCGATCCACGGAGGGAAG"
    string1 = "GCACATAGCCGCGCTATCCGACAATCTCCAAATTATAACATACCGTTCCATGAAGGCCAGAATTACTTACCGGCCCTTTCCATGCGTGCGCCATACCCCCCCACTCCCCCGCTTATCCGTCCGAGGGGAGAGTGTGCGATCCTCCGTTAAGATATTCTTACGTATGACGTAGCTATGTATTTTGCAGAGGTAGCGAACGCGTTGAACACTTCACAGATGGTGGGGATTCGGGCAAAGGGCGTATAATTGGGGACTAACATAGGCGTAAACTACGATGGCACCAACTCAATCGCAGCTCGTGCGCCCTGAATAACGTACTCATCTCAACTGATTCTCGGCAATCTACGGAGCGACTTGATTATCAACAGCTGTCTAGCAGTTCTAATCTTTTGCCAACATCGTAATAGCCTCCAAGAGATTGATCATACCTATCGGCACAGAAGTGACACGACGCCGATGGGTAGCGGACTTTTGGTCAACCACAATTCCCCAGGGGACAGGTCCTGCGGTGCGCATCACTTTGTATGTGCAAGCAACCCAAGTGGGCCCAGCCTGGACTCAGCTGGTTCCTGTGTGAGCTCGAGGCTGGGGATGACAGCTCTTTAAACATAGGGCGGGGGCGTCGAACGGTCGAGAAACTCATAGTACCTCGGGTACCAACTTACTCAGGTTATTGCTTGAAGCTGTACTATTTCAGGGGGGGAGCGCTGAAGGTCTCTTCTTCTGATGACTGAACTCGCAAGGGTCGTGAAGTCGGTTCCTTCAATGGTTAAAAAACAAAGGCTTACTGTGCAGACTGGAACGCCCATCTAGCGGCTCGCGTCTTGAATGCTCGGTCCCCTTTGTCATTCCGGATAAATCCATTTCCCTCATTCACCAGCTTGCGAAGTCTACATTGGTATATGAATGCGACCTAGAAGAGGGCGCTTAAAATTGGGAGTGGTTGATGCTCTATACTCCATTTGGTTTTTTCGTGCATCACCGCGATAGGCTGACAAGG"
    string2 = "CCCGACCTCACATAGCCGCGCTATCCGACAATCTCCAAATTATAACATACCGTTCCATGAAGGCCAGAATTACTTACCGGCCCTTTCCATGCGTGCGCCATACCCCCCCACTTTGTACAGGGTACCATCGGGATTCTGAACCCTCAGATAGTGGGGAGGTGGGGATTCGGGCAAAGGGCGTATAATTGGGGACTAACATAGGCGTAAACTACGATGGCACCAACTCAATCGCAGCTCGTGCGCCCTGAATAACGTACTCATCTCAACTGATTCTCGGCAATCTACGGAGCGACTTACGCCGCCACGTGTTCGTTAACTGTTGATTGGTAGCACAAAAGTAATACCATGGTCCTTGAAATTCGGCTCAGTTAGATTCTCGGCAATCTACGGAGCGACTTGATTATCAACAGCTGTCTAGCGAAACGCGCCCAAGTGACGCTAGGCAAGTCAGAGCAGGTTCCCGTGTTAGCTTAAGGGTAAACATACAAGTCGATTGAAGATGGGTAGGGGGCTTCAATTCGTCCAGCACTCTACGGTACCTCCGAGAGCAAGTAGGGCACCCTGTAGTTCGAAGCGGAACTATTTCGTGGGGCGAGCCCACATCGTCTCTTCTGCGGATGACTTAACACGTTAGGGAGGCTGGGGATGACAGCTCTTTAAACATAGGGCGGGGGCGTCGAACGGTCGAGAAACTCATAGTACCTCGGGTACAACGGTGCGTAACTCGATCACTCACTCGCAGCGCTCATACACTTGGTTCCGAGGCCTGTCCTGATATATGAACCCAAACTAGAGCGGGGCTGTTGACGTTTGGAGTTGAAAAAATCTAATATTCCAATCGGCTTCAACGTGCACCACCGCAGGCGGCTGCCCTCATTCACCAGCTACGAGGGGCTCACACCGAGATAGAAGAGGCGCTTAAAATTGGGAGTGGTTGATGCTCTATACTCCATTTGGTTTTTAAGTAGACTGTTGCGCGTTGGGGGTAGCGCCGGCTAACAAAG"
    #string1 = "CCTTGCTACGCACGGGCACGGAGCGCAGCCCCAGCCACCCCTAATCACACACCTTGCTACGCACGGGCACGGAGCGCAGCCCCAGCCACCCCTAATCACACACCTTGCTACGCACGGGCACGGAGCGCAGCCCCAGCCACCCCTAATCACACA"
    #string2 = "CCGGCCCCGGAACGGTTTGCACCTGGGAATCAGCGCCGCCTCGGCCGATAACCTTGCTACGCACGGGCACGGAGCGCAGCCCCAGCCACCCCTAATCACACACCTTGCTACGCACGGGCACGGAGCGCAGCCCCAGCCACCCCTAATCACACA"
    #string1 = "CCTGGCTAC"
    #string2 = "CCGGCCTAAC"
    #string1 = "CCTG"
    #string2 = "CCGGCC"
    seq1 = np.fromstring(string1, dtype='|S1')
    seq2 = np.fromstring(string2, dtype='|S1')
    
    len1 = len(seq1)
    len2 = len(seq2)
    print('Sequence length: {}, {} '.format(len1, len2) )
    
    gpu_seq1_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY, len1)
    gpu_seq2_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY, len2)
    
    # Send to the device, non-blocking
    cl.enqueue_copy(queue, gpu_seq1_buff, seq1, is_blocking=False)
    cl.enqueue_copy(queue, gpu_seq2_buff, seq2, is_blocking=False)
    queue.finish()
    
    # set different local size to test, this number also specifies the number of workers.
    local_size = (32, 32)
    global_size = tuple([round_up(g, l) for g, l in zip((len1, len2), local_size)])
    print global_size
    print local_size
    
    # two partial DP tables: row and column are both stored in 1D array in order to coalesced read/write
    row_table = np.zeros( (len1+1)*(global_size[1]/local_size[1]+1), dtype=np.uint32)
    col_table = np.zeros( (global_size[0]/local_size[0]+1)*(len2+1), dtype=np.uint32)
    row_dptable = cl.Buffer(context, cl.mem_flags.READ_WRITE, row_table.size *4)
    col_dptable = cl.Buffer(context, cl.mem_flags.READ_WRITE, col_table.size *4)
    
    width = np.int32(len1+1)
    height = np.int32(len2+1)
    edge = np.int32(1)
    
    # Create a local memory per working group that is
    # the size of an int (4 bytes) * (M+1) * (N+1), where M,N are the local_size{0] and [1]
    buf_size = (np.int32(local_size[0] + edge), np.int32(local_size[1] + edge))
    gpu_local_memory = cl.LocalMemory(4 * buf_size[0] * buf_size[1])

    # initialize labels
    program.initialize_table(queue, global_size, local_size,
                             row_dptable, col_dptable, width, height)

    # count number of blocks, it determines how many parallel block we can have.
    itercount = np.int32(global_size[0]/local_size[0] + global_size[1]/local_size[1])
    show_progress = False

    # Show the initial labels
    cl.enqueue_copy(queue, row_table, row_dptable, is_blocking=True)
    cl.enqueue_copy(queue, col_table, col_dptable, is_blocking=True)
    # print "Initialization"
    # print row_table.reshape([-1,len1+1])
    # print col_table.reshape([-1,len2+1])
    
    total_time = 0

    for itr in xrange(itercount):
        prop_exec = program.needleman_coalesce(queue, global_size, local_size,
                                               gpu_seq1_buff, gpu_seq2_buff,
                                               row_dptable,
                                               col_dptable,
                                               gpu_local_memory,
                                               np.int32(itr),
                                               width, height,
                                               buf_size[0], buf_size[1],
                                               edge )
        prop_exec.wait()
        elapsed = 1e-6 * (prop_exec.profile.end - prop_exec.profile.start)
        total_time += elapsed
        if show_progress:
            print "itr"
            cl.enqueue_copy(queue, row_table, row_dptable, is_blocking=True)
            cl.enqueue_copy(queue, col_table, col_dptable, is_blocking=True)
            print row_table.reshape([-1,len1+1])
            print col_table.reshape([-1,len2+1])

    # Show final result
    cl.enqueue_copy(queue, row_table, row_dptable, is_blocking=True)
    cl.enqueue_copy(queue, col_table, col_dptable, is_blocking=True)
    
    printresult = False
    
    row_table = row_table.reshape([-1,len1+1])
    col_table = col_table.reshape([-1,len2+1])
    if printresult:
        print "Parallel result:"
        print "row"
        print row_table
        print "column"
        print col_table

    s_time = time.time()
    serial = edit_distance(string1, string2)
    s_time = time.time() - s_time
    s_time *= 1e3
    if printresult:
        print "Serial result:"
        print serial
        
    assert (row_table[-1,:] == serial[-1,:]).all()
    assert (col_table[-1,:].T == serial[:,-1]).all()
    print('Edit distance: {}'.format(row_table[-1,-1]) )
    print('Parallel time: {} ms'.format(total_time) )
    print('Serial time: {} ms'.format(s_time) )
    print('Speed Ratio: {}'.format(s_time/total_time))
    

