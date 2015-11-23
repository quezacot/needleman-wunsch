from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import pylab
import string
import time

# Edit distance = 1 for insertion, deletion, and replacement
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
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('nw.cl').read()).build(options='')

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
    string1 = "CCTTGCTACGCACGGGCACGGAGCGCAGCCCCAGCCACCCCTAATCACACA"
    string2 = "CCGGCCCCGGAACGGTTTGCACCTGGGAATCAGCGCCGCCTCGGCCGATAA"
    seq1 = np.fromstring(string1, dtype='|S1')
    seq2 = np.fromstring(string2, dtype='|S1')
    
    len1 = len(seq1)+1
    len2 = len(seq2)+1
    sz = min(len1, len2)
    
    host_table = np.zeros(len1*len2, dtype=np.uint32)
    #print host_table.reshape([len2,len1])
    gpu_seq1_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY, len1)
    gpu_seq2_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY, len2)
    dptable = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_table.size *4)
    #gpu_done_flag = cl.Buffer(context, cl.mem_flags.READ_WRITE, 4)
    

    # Send to the device, non-blocking
    cl.enqueue_copy(queue, gpu_seq1_buff, seq1, is_blocking=False)
    cl.enqueue_copy(queue, gpu_seq2_buff, seq2, is_blocking=False)
    queue.finish()
    
    local_size = (1, sz)  # ?? per work group
    global_size = (len1, len2)
    #print global_size
    width = np.int32(len1)
    height = np.int32(len2)
    
    # Create a local memory per working group that is
    # the size of an int (4 bytes) * (N+2) * (N+2), where N is the local_size
    #buf_size = (np.int32(local_size[0] + 2 * halo), np.int32(local_size[1] + 2 * halo))
    #gpu_local_memory = cl.LocalMemory(4 * buf_size[0] * buf_size[1])

    # initialize labels
    program.initialize_table(queue, global_size, local_size,
                             dptable, width, height)

    # while not done, propagate labels
    itercount = len1 + len2 - 1
    show_progress = False

    # Show the initial labels
    cl.enqueue_copy(queue, host_table, dptable, is_blocking=True)
    #print host_table.reshape([len2,len1])
    
    total_time = 0

    for itr in xrange(2,itercount):
        #host_done_flag[0] = 0
        #cl.enqueue_copy(queue, gpu_done_flag, host_done_flag, is_blocking=False)
        prop_exec = program.needleman_by1(queue, global_size, local_size,
                                          gpu_seq1_buff, gpu_seq2_buff,
                                          dptable,
                                          np.int32(itr),
                                          width, height)
        prop_exec.wait()
        elapsed = 1e-6 * (prop_exec.profile.end - prop_exec.profile.start)
        total_time += elapsed
        if show_progress:
            cl.enqueue_copy(queue, host_table, dptable, is_blocking=True)
            print host_table.reshape([len2,len1])
        print ""
    # Show final result
    cl.enqueue_copy(queue, host_table, dptable, is_blocking=True)
    
    s_time = time.time()
    serial = edit_distance(string1, string2)
    s_time = time.time() - s_time
    assert (host_table.reshape([len2,len1]) == serial).all()
    #print host_table.reshape([len2,len1])
    #print edit_distance(string1, string2)
    print('Parallel time: {}'.format(total_time) )
    print('Serial time: {}'.format(s_time) )
    

