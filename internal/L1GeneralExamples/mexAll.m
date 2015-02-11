
% minFunc
fprintf('Compiling minFunc files...\n');
mex minFunc_2012/mex/mcholC.c -outdir minFunc_2012/compiled
mex minFunc_2012/mex/lbfgsC.c -outdir minFunc_2012/compiled

fprintf('Compiling repmatC...\n');
mex -IKPM KPM/repmatC.c -outdir KPM

% UGM
fprintf('Compiling UGM files...\n');
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_makeEdgeVEC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_ExactC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_ExactC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_ChainC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_makeClampedPotentialsC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_ICMC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_GraphCutC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Sample_GibbsC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_MFC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_LBPC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_LBPC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Infer_TRBPC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_TRBPC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_CRF_makePotentialsC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_CRF_PseudoNLLC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_LogConfigurationPotentialC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_AlphaExpansionC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_Decode_AlphaExpansionBetaShrinkC.c
mex -IUGM_2011/mex -outdir UGM_2011/compiled UGM_2011/mex/UGM_CRF_NLLC.c