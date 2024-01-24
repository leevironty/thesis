#! /bin/bash -l

ml restore lintim
ml load miniconda

conda activate lintim-38

CLASSPATH=${GUROBI_HOME}/lib/gurobi.jar:${CLASSPATH}
LD_LIBRARY_PATH=${GUROBI_HOME}/lib/:${LD_LIBRARY_PATH}

cd $1 && make tim-timetable
