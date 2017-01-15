import tensorflow as tf
from MelAPI.ctype_util import *

# TODO: fill out the rest of this table
ctypes2TF = {
    c_bool: tf.bool,
    c_float: tf.float32,
    c_double: tf.float64,
    c_uint: tf.int64,  # no tf.uint32 :(
}


def inputCType(ctype, shape=None, name=""):
    if ctype in ctypes2TF:
        return tf.placeholder(ctypes2TF[ctype], shape, name)
    elif issubclass(ctype, Structure):
        return {f: inputCType(t, shape, name + "/" + f) for (f, t) in ctype._fields_}
    else:  # assume an array type
        base_type = ctype._type_
        return [inputCType(base_type, shape, name + "/" + str(i)) for i in range(ctype._length_)]

def constantCTypes(ctype, values, name=""):
  if ctype in ctypes2TF:
    return tf.constant(values, dtype=ctypes2TF[ctype], name=name)
  elif issubclass(ctype, Structure):
    return {f : constantCTypes(t, [getattr(v, f) for v in values], name + "/" + f) for (f, t) in ctype._fields_}
  else: # assume an array type
    base_type = ctype._type_
    return [inputCType(base_type, [v[i] for v in values], name + "/" + str(i)) for i in range(ctype._length_)]

def feedCType(ctype, name, value, feed_dict=None):
  if feed_dict is None:
    feed_dict = {}
  if ctype in ctypes2TF:
    feed_dict[name + ':0'] = value
  elif issubclass(ctype, Structure):
    for f, t in ctype._fields_:
      feedCType(t, name + '/' + f, getattr(value, f), feed_dict)
  else: # assume an array type
    base_type = ctype._type_
    for i in range(ctype._length_):
      feedCType(base_type, name + '/' + str(i), value[i], feed_dict)

  return feed_dict

def feedCTypes(ctype, name, values, feed_dict=None):
  if feed_dict is None:
    feed_dict = {}
  if ctype in ctypes2TF:
    feed_dict[name + ':0'] = values
  elif issubclass(ctype, Structure):
    for f, t in ctype._fields_:
      feedCTypes(t, name + '/' + f, [getattr(v, f) for v in values], feed_dict)
  else: # assume an array type
    base_type = ctype._type_
    for i in range(ctype._length_):
      feedCTypes(base_type, name + '/' + str(i), [v[i] for v in values], feed_dict)

  return feed_dict

def vectorizeCTypes(ctype, values):
  if ctype in ctypes2TF:
    return np.array(values)
  elif issubclass(ctype, Structure):
    return {f : vectorizeCTypes(t, [getattr(v, f) for v in values]) for (f, t) in ctype._fields_}
  else: # assume an array type
    base_type = ctype._type_
    return [vectorizeCTypes(base_type, [v[i] for v in values]) for i in range(ctype._length_)]
