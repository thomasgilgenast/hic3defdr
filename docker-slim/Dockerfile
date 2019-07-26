FROM creminslab/lib5c:0.5.4-slim

# python deps
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# application code
ARG VERSION=unknown
COPY dist/hic3defdr-$VERSION-py2-none-any.whl .
RUN pip install hic3defdr-$VERSION-py2-none-any.whl

ENTRYPOINT bash
