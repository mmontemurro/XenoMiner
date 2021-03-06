include: '../snakemake/conf.sk'


rule all:
    input: 
            expand(DATASET + "/{assembly}_{len}/{chrom}_{len}_{k}mer_freq", assembly=[ASSEMBLY, ASSEMBLY_M], len=150, chrom="chr1", k=12),
            expand(DATASET + "/datasets/{chrom}_{len}_{k}mer.dat", chrom="chr1", len=150, k=12),
            expand(DATASET + "/datasets/{chrom}_{len}_{k}mer.lab",  chrom="chr1", len=150, k=12),
            expand(DATASET + "/cnn-{chrom}_{len}_{k}mer_bin_cross_entr2.done", chrom="chr1", len=150, k=12)
           #expand(DATA_DIR + "/kmers_stats/GRCh38_GRCm38_{k}mer_pearson_corr", k=12), 
           #expand(DATA_DIR + "/" + ASSEMBLY + "_{len}/{chrom}_{len}_frags", chrom=CHROMOSOMES, len=READLEN),
           #expand(DATA_DIR + "/{k}mer_dictionary", k=K),
           #expand(DATA_DIR + "/" + ASSEMBLY + "_{len}/{chrom}_{len}_{k}mer_freq", chrom=["chr1"], len=READLEN, k=K),
           #expand(DATA_DIR + "/kmers_stats/"+ASSEMBLY+"_{chrom}_{k}mer_count", k=12, chrom=CHROMOSOMES),
           
           

#filter chromosomes (keep chr1-22,X,Y)
#samtools faidx {input} chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY > local/share/data/GRCh38.fa

rule split_chrom: #Split reference genome fasta by chromosome
    input:ASSEMBLY_FA
    output:expand(CHROM_DIR + "/{chrom}.fa", chrom=CHROMOSOMES)
    params: outdir=CHROM_DIR
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi

            faSplit byname  {input} {params.outdir}
        """

#rule kmers_count:
#    input: fasta_file=CHROM_DIR + "/{chrom}.fa", kmer_list=DATA_DIR + "/{k}mer_dictionary"
#    output: DATA_DIR + "/kmers_stats/"+ASSEMBLY+"_{chrom}_{k}mer_count"
#    params: outdir=DATA_DIR + "/kmers_stats/", tool=FASTA2MATRIX, k="{k}"
#    log: LOGS_DIR + "/kmers_count-" + ASSEMBLY + "_{chrom}_{k}mer.log"
#    benchmark: BENCHMARKS_DIR + "/kmers_count-" + ASSEMBLY + "_{chrom}_{k}mer.tsv"
#    shell:
#        """
#            if [ ! -d {params.outdir} ]; then
#                mkdir -p  {params.outdir}
#            fi
#
#            python {params.tool} -k {params.k} -i {input.fasta_file} -d {input.kmer_list} -o {output} 2> {log}
#        """

#rule kmers_corr:
#    input: h = expand(DATA_DIR + "/kmers_stats/"+ASSEMBLY+"_{chrom}_{k}mer_count", k=12, chrom=CHROMOSOMES),
#            m = expand(DATA_DIR + "/kmers_stats/"+ASSEMBLY_M+"_{chrom}_{k}mer_count", k=12, chrom=CHROMOSOMES_M)
#    output:
#        DATA_DIR + "/kmers_stats/GRCh38_{k}mer_count",
#        DATA_DIR + "/kmers_stats/GRCm38_{k}mer_count", 
#        DATA_DIR + "/kmers_stats/GRCh38_GRCm38_{k}mer_pearson_corr"
#    params: tool=KMERSTATS,
#            k=12,
#            outdir=DATA_DIR + "/kmers_stats"
#    log: LOGS_DIR + "/{k}mers_corr.log"
#    benchmark: BENCHMARKS_DIR + "{k}mers_corr.tsv"
#    shell:
#        """
#            python {params.tool} -H {input.h} -M {input.m} -K {params.k} -O {params.outdir} 2> {log}
#        """
         

#rule split_genome: #Simulating $len bp Reads
#    input: CHROM_DIR + "/{chrom}.fa"
#    output: DATA_DIR + "/" + ASSEMBLY + "_{len}/{chrom}_{len}_frags"
#    params: tool=BIN_DIR+"/simReads", outdir=DATA_DIR + "/" + ASSEMBLY + "_{len}"
#    shell:
#        """
        #if [ ! -d {params.outdir} ]; then
        #    mkdir -p {params.outdir}
        #fi

        #{params.tool} {input} {output} {wildcards.len}
#        """
#rule make_kmer_list: #generating dictionary of kmers for different values of K
#    output: DATA_DIR + "/{k}mer_dictionary"
#    params: alphabet=ALPHABET, tool=MK_KMERS_DICT, outdir=DATA_DIR
#    shell:
#        """
        #python {params.tool} -k {wildcards.k} -a {params.alphabet} -o {params.outdir}
#        """



rule kmers_frequency:
    input: kmers_list = DATA_DIR + "/{k}mer_dictionary_dsk", fasta_file = DATA_DIR + "/{assembly}_{len}/{chrom}_{len}_frags"
    output: DATASET + "/{assembly}_{len}/{chrom}_{len}_{k}mer_freq"
    log: LOGS_DIR + "/kmers_frequency-{assembly}_{chrom}_{len}_{k}merlog"
    benchmark: BENCHMARKS_DIR + "/kmers_frequency-{assembly}_{chrom}_{len}_{k}mertsv"
    params: tool=FASTA2MATRIX, k="{k}", outdir=DATASET + "/{assembly}_{len}/"
    shell:
        """
        if [ ! -d {params.outdir} ]; then
            mkdir -p {params.outdir}
        fi

        python {params.tool} -k {params.k} -i {input.fasta_file} -d {input.kmers_list} -o {output} -g -f 2> {log}
        """

rule label:
    input: h=expand(DATASET + "/{assembly}_{{len}}/{{chrom}}_{{len}}_{{k}}mer_freq", assembly=ASSEMBLY),
            m=expand(DATASET + "/{assembly}_{{len}}/{{chrom}}_{{len}}_{{k}}mer_freq", assembly=ASSEMBLY_M)
    output: DATASET + "/datasets/{chrom}_{len}_{k}mer.dat", DATASET + "/datasets/{chrom}_{len}_{k}mer.lab"
    params: tool=LABELLING,  outdir=DATASET + "/datasets", outprefix=DATASET + "/datasets/{chrom}_{len}_{k}mer"
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi

            python {params.tool} -hu {input.h} -m {input.m} -o {params.outprefix}
        """

rule cnn:
    input: DATASET + "/datasets/{chrom}_{len}_{k}mer.dat", DATASET + "/datasets/{chrom}_{len}_{k}mer.lab"
    output: touch(DATASET + "/cnn-{chrom}_{len}_{k}mer_bin_cross_entr2.done")
    params: in_prefix = DATASET + "/datasets/{chrom}_{len}_{k}mer", out_prefix = DATASET + "/models/{chrom}_{len}_{k}mer_bin_cross_entr2", 
            outdir = DATASET + "/models", threads = 20, cnn = CNN
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi

            python {params.cnn} -i {params.in_prefix} -o {params.out_prefix} -t {params.threads}
        """
