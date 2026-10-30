"""Python 3.6: explicit invalid-UV zero sampling instead of CPU out-of-range gather."""
from __future__ import print_function
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


OLD='    uv_map = tf.gather_nd(img_attrs, batch_uv_pos)'
NEW='''    # Research CPU compatibility: invalid UVs are not observations.
    limits = tf.shape(img_attrs)[:3]
    valid_uv = tf.reduce_all(tf.logical_and(batch_uv_pos >= 0, batch_uv_pos < limits), axis=-1)
    safe_uv = tf.minimum(tf.maximum(batch_uv_pos, 0), limits - 1)
    uv_map = tf.gather_nd(img_attrs, safe_uv)
    uv_map = uv_map * tf.cast(valid_uv[..., None], uv_map.dtype)'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    with tf.Graph().as_default():
        img_attrs=tf.constant(np.arange(6,dtype=np.float32).reshape(1,2,3,1))
        batch_uv_pos=tf.constant([[[0,1,2],[0,-1,-1],[0,20,20]]])
        scope={'tf':tf,'img_attrs':img_attrs,'batch_uv_pos':batch_uv_pos}
        exec('\n'.join(line[4:] for line in NEW.splitlines()),scope)
        grad=tf.gradients(tf.reduce_sum(scope['uv_map']),img_attrs)[0]
        with tf.Session() as session:value,gradient=session.run([scope['uv_map'],grad])
        np.testing.assert_array_equal(value.ravel(),[5,0,0])
        np.testing.assert_array_equal(gradient.ravel(),[0,0,0,0,0,1])
    path=a.source/'utils/unwrap_utils.py';text=path.read_text()
    if text.count(OLD)!=1:raise ValueError('Author source changed or already patched')
    before=hashlib.sha256(path.read_bytes()).hexdigest();backup=path.with_suffix('.py.research-original')
    if backup.exists():raise FileExistsError(str(backup))
    path.rename(backup);path.write_text(text.replace(OLD,NEW),encoding='utf-8')
    report=dict(source='utils/unwrap_utils.py',before_sha256=before,after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        contract_values=value.tolist(),contract_gradient=gradient.tolist(),scope='invalid UVs zeroed, valid samples and gradients preserved')
    with a.report.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print('UV bounds and gradient contract passed')


if __name__=='__main__':main()
