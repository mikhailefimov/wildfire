FROM kaggle/python

RUN rm -rf /opt/conda/lib/python3.6/site-packages/tensorflow && \
    pip uninstall -y tensorflow && \
    conda install -y tensorflow==1.11 && \
    rm -rf /root/.cache/pip/* && \
    conda clean -y --tarballs

RUN conda install -c conda-forge -y gdal rtree geopandas fiona  && \
    conda clean -y --tarballs