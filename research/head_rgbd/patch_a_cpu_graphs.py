"""Python 3.6: wrap frozen NCHW ops with equivalent CPU NHWC transposes.

Only task-owned extracted PB copies are changed. Original bytes are retained.
No weights are learned or substituted. Unknown NCHW operators stop conversion.
"""
from __future__ import print_function
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import node_def_pb2


SUPPORTED={'Conv2D','DepthwiseConv2dNative','Conv2DBackpropInput','MaxPool','AvgPool','BiasAdd'}


def convert(graph):
    result=tf.GraphDef();changes=[]
    for original in graph.node:
        if original.attr.get('data_format') is None or original.attr['data_format'].s!=b'NCHW':
            result.node.add().CopyFrom(original);continue
        if original.op not in SUPPORTED:raise ValueError('Unsupported NCHW operator: '+original.op+' '+original.name)
        name=original.name;dtype=original.attr['T'].type
        for suffix,perm in [('to_nhwc',[0,2,3,1]),('to_nchw',[0,3,1,2])]:
            const=result.node.add();const.name=name+'/'+suffix+'_perm';const.op='Const'
            const.attr['dtype'].type=tf.int32.as_datatype_enum
            const.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(perm,dtype=tf.int32))
        trans=result.node.add();trans.name=name+'/to_nhwc';trans.op='Transpose'
        input_index=2 if original.op=='Conv2DBackpropInput' else 0
        trans.input.extend([original.input[input_index],name+'/to_nhwc_perm'])
        trans.attr['T'].type=dtype;trans.attr['Tperm'].type=tf.int32.as_datatype_enum
        internal=result.node.add();internal.CopyFrom(original);internal.name=name+'/cpu_nhwc'
        internal.input[input_index]=name+'/to_nhwc';internal.attr['data_format'].s=b'NHWC'
        if original.op=='Conv2DBackpropInput':
            size=result.node.add();size.name=name+'/nhwc_output_shape';size.op='Gather'
            size.input.extend([original.input[0],name+'/to_nhwc_perm'])
            size.attr['Tparams'].type=tf.int32.as_datatype_enum
            size.attr['Tindices'].type=tf.int32.as_datatype_enum
            size.attr['validate_indices'].b=True
            internal.input[0]=size.name
        if '_output_shapes' in internal.attr:del internal.attr['_output_shapes']
        for key in ['strides','ksize','dilations']:
            if key in internal.attr:
                values=list(internal.attr[key].list.i)
                if len(values)!=4:raise ValueError('Unexpected '+key)
                del internal.attr[key].list.i[:]
                internal.attr[key].list.i.extend([values[i] for i in [0,2,3,1]])
        if 'explicit_paddings' in internal.attr and len(internal.attr['explicit_paddings'].list.i):
            raise ValueError('Explicit padding requires separately reviewed conversion')
        output=result.node.add();output.name=name;output.op='Transpose'
        output.input.extend([internal.name,name+'/to_nchw_perm'])
        output.attr['T'].type=dtype;output.attr['Tperm'].type=tf.int32.as_datatype_enum
        changes.append(dict(name=name,op=original.op))
    return result,changes


def contract_test():
    graph=tf.Graph()
    rng=np.random.RandomState(17);x=rng.normal(size=(1,7,9,2)).astype(np.float32)
    w=rng.normal(size=(3,3,2,3)).astype(np.float32)
    with graph.as_default():
        source=tf.constant(x,name='input');kernel=tf.constant(w,name='kernel')
        nchw=tf.transpose(source,[0,3,1,2])
        conv=tf.nn.conv2d(nchw,kernel,[1,1,1,1],padding='SAME',data_format='NCHW')
        pool=tf.nn.max_pool(conv,[1,1,2,2],[1,1,2,2],padding='SAME',data_format='NCHW')
        tf.nn.conv2d_transpose(pool,kernel,[1,2,8,10],[1,1,2,2],padding='SAME',data_format='NCHW',name='output')
    patched,changes=convert(graph.as_graph_def())
    target=tf.Graph()
    with target.as_default():tf.import_graph_def(patched,name='')
    with tf.Session(graph=target) as session:actual=session.run(target.get_tensor_by_name('output:0'))
    reference=tf.Graph()
    with reference.as_default():
        c=tf.nn.conv2d(tf.constant(x),tf.constant(w),[1,1,1,1],padding='SAME')
        pooled=tf.nn.max_pool(c,[1,2,2,1],[1,2,2,1],padding='SAME')
        deconv=tf.nn.conv2d_transpose(pooled,tf.constant(w),[1,8,10,2],[1,2,2,1],padding='SAME')
    with tf.Session(graph=reference) as session:expected=session.run(deconv)
    np.testing.assert_allclose(actual.transpose(0,2,3,1),expected,atol=1e-5,rtol=1e-5)
    return float(np.max(np.abs(actual.transpose(0,2,3,1)-expected)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pb-directory',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();report=dict(contract_max_absolute_error=contract_test(),models=[])
    for path in sorted(a.pb_directory.glob('*.pb')):
        if path.name.startswith('pix2pix'):continue
        graph=tf.GraphDef();graph.ParseFromString(path.read_bytes())
        converted,changes=convert(graph)
        row=dict(path=str(path),changes=changes,original_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if changes:
            backup=path.with_suffix('.pb.original')
            if backup.exists():raise FileExistsError(str(backup))
            path.rename(backup);path.write_bytes(converted.SerializeToString())
        row['converted_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();report['models'].append(row)
        print(path.name+': '+str(len(changes))+' operations converted',flush=True)
    with a.report.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print('Layout conversion contract passed:',report['contract_max_absolute_error'])


if __name__=='__main__':main()
