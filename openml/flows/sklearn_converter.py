"""Convert scikit-learn estimators into an OpenMLFlows and vice versa."""

from collections import OrderedDict
import copy
from distutils.version import LooseVersion
import importlib
import inspect
import json
import json.decoder
import keras
import re
import six
import warnings
import sys
import inspect


import numpy as np
import scipy.stats.distributions
import sklearn.base
import sklearn.model_selection
# Necessary to have signature available in python 2.7
from sklearn.utils.fixes import signature
from keras.layers import Dense, Dropout

import openml
from openml.flows import OpenMLFlow
from openml.exceptions import PyOpenMLError

DEBUG=True

if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError


DEPENDENCIES_PATTERN = re.compile(
    '^(?P<name>[\w\-]+)((?P<operation>==|>=|>)(?P<version>(\d+\.)?(\d+\.)?(\d+)?(dev)?[0-9]*))?$')

def tes_serialize_seq(o):
    dummydict=OrderedDict([('description',None),('data_type',None)]  )
    a,b,c,d=_extract_information_from_model(o)
    for key in b['build_fn'].keys():
        print(key)
        #c[key]=dummydict
    return a,b,c,d
    
def sklearn_to_flow(o, parent_model=None):
    # TODO: assert that only on first recursion lvl `parent_model` can be None

    if _is_sequential_wrapper(o):
        #o.sk_params['model']=o.build_fn().get_config()
        #print(o.sk_params)
        # TODO: explain what type of parameter is here
        rval = _serialize_model(o) 

    elif _is_estimator(o):
        # is the main model or a submodel
        rval = _serialize_model(o)
    

    elif _is_sequential_function(o):
        # TODO: explain what type of parameter is here
        #rval = _serialize_sequential_model(o)
        submodel=o()
        #tempfn=o['build_fn']
        #'layer'+str(layer_id)
        
        #rval=[() for i in submodel.get_config()]
        #rval['steps']={}
        #for layer_id,layer in enumerate(submodel.get_config()):
        #    if(layer['class_name']=='Dense'):
        #        rval[layer_id]=('dense',Dense.from_config(layer['config']))
        #    elif(layer['class_name']=='Dropout'):
        #        rval[layer_id]=('dropout',Dropout.from_config(layer['config']))
        #tp=tuple(['steps',rval])
        ###rval= sklearn_to_flow(translate_seq(submodel),parent_model)
        return (inspect.getsource(o))[:2000]
        #rval= _serialize_sequential_model(submodel)

    elif _is_layer(o):
        rval = _serialize_sequential_model(o) 
        #rval=sklearn_to_flow(o.get_config())
        #return rval
    elif isinstance(o, (list, tuple)):
        # TODO: explain what type of parameter is here
        rval = [sklearn_to_flow(element, parent_model) for element in o]
        if isinstance(o, tuple):
            rval = tuple(rval)
    elif isinstance(o, (bool, int, float, six.string_types)) or o is None:
        # base parameter values
        rval = o
    elif isinstance(o, dict):
        # TODO: explain what type of parameter is here
        if not isinstance(o, OrderedDict):
            o = OrderedDict([(key, value) for key, value in sorted(o.items())])

        rval = OrderedDict()
        for key, value in o.items():
            if not isinstance(key, six.string_types):
                raise TypeError('Can only use string as keys, you passed '
                                'type %s for value %s.' %
                                (type(key), str(key)))
            key = sklearn_to_flow(key, parent_model)
            value = sklearn_to_flow(value, parent_model)
            rval[key] = value
        rval = rval
    elif isinstance(o, type):
        # TODO: explain what type of parameter is here
        rval = serialize_type(o)
    elif isinstance(o, scipy.stats.distributions.rv_frozen):
        rval = serialize_rv_frozen(o)
    # This only works for user-defined functions (and not even partial).
    # I think this is exactly what we want here as there shouldn't be any
    # built-in or functool.partials in a pipeline
    elif inspect.isfunction(o):
        # TODO: explain what type of parameter is here
        print("fungsi")
        rval = serialize_function(o)
    elif _is_cross_validator(o):
        # TODO: explain what type of parameter is here
        rval = _serialize_cross_validator(o)
    
    else:
        raise TypeError(o, type(o))

    return rval


def _is_estimator(o):
    return (hasattr(o, 'fit') and hasattr(o, 'get_params') and
            hasattr(o, 'set_params'))


def _is_cross_validator(o):
    return isinstance(o, sklearn.model_selection.BaseCrossValidator)

def _is_sequential_function(o):
    return inspect.isfunction(o) and (isinstance(o(), keras.models.Sequential) or isinstance(o(), keras.models.Model))

def _is_sequential_wrapper(o):
    return isinstance(o, keras.wrappers.scikit_learn.KerasClassifier)

def _is_layer(o):
    try:
        return 'keras.layers' in str(o.__class__)
    except Exception:
        return false



def flow_to_sklearn(o, **kwargs):
    # First, we need to check whether the presented object is a json string.
    # JSON strings are used to encoder parameter values. By passing around
    # json strings for parameters, we make sure that we can flow_to_sklearn
    # the parameter values to the correct type.
    if isinstance(o, six.string_types):
        try:
            o = json.loads(o)
        except JSONDecodeError:
            pass

    if isinstance(o, dict):
        # Check if the dict encodes a 'special' object, which could not
        # easily converted into a string, but rather the information to
        # re-create the object were stored in a dictionary.
        if 'oml-python:serialized_object' in o:
            serialized_type = o['oml-python:serialized_object']
            value = o['value']
            if serialized_type == 'type':
                rval = deserialize_type(value, **kwargs)
            elif serialized_type == 'rv_frozen':
                rval = deserialize_rv_frozen(value, **kwargs)
            elif serialized_type == 'function':
                rval = deserialize_function(value, **kwargs)
            elif serialized_type == 'component_reference':
                value = flow_to_sklearn(value)
                step_name = value['step_name']
                key = value['key']
                component = flow_to_sklearn(kwargs['components'][key])
                # The component is now added to where it should be used
                # later. It should not be passed to the constructor of the
                # main flow object.
                del kwargs['components'][key]
                if step_name is None:
                    rval = component
                else:
                    rval = (step_name, component)
            elif serialized_type == 'cv_object':
                rval = _deserialize_cross_validator(value, **kwargs)
            else:
                raise ValueError('Cannot flow_to_sklearn %s' % serialized_type)

        else:
            rval = OrderedDict((flow_to_sklearn(key, **kwargs),
                                flow_to_sklearn(value, **kwargs))
                               for key, value in sorted(o.items()))
    elif isinstance(o, (list, tuple)):
        rval = [flow_to_sklearn(element, **kwargs) for element in o]
        if isinstance(o, tuple):
            rval = tuple(rval)
    elif isinstance(o, (bool, int, float, six.string_types)) or o is None:
        rval = o
    elif isinstance(o, OpenMLFlow):
        rval = _deserialize_model(o, **kwargs)
    #elif isinstance(o, keras.models.Sequential):
    #    rval = _deserialize_seq_model(o, **kwargs)    
    else:
        raise TypeError(o)

    return rval


def _serialize_model(model):
    """Create an OpenMLFlow.

    Calls `sklearn_to_flow` recursively to properly serialize the
    parameters to strings and the components (other models) to OpenMLFlows.

    Parameters
    ----------
    model : sklearn estimator

    Returns
    -------
    OpenMLFlow

    """

    # Get all necessary information about the model objects itself
    parameters, parameters_meta_info, sub_components, sub_components_explicit =\
        _extract_information_from_model(model)

    # Check that a component does not occur multiple times in a flow as this
    # is not supported by OpenML
    _check_multiple_occurence_of_component_in_flow(model, sub_components)

    # Create a flow name, which contains all components in brackets, for
    # example RandomizedSearchCV(Pipeline(StandardScaler,AdaBoostClassifier(DecisionTreeClassifier)),StandardScaler,AdaBoostClassifier(DecisionTreeClassifier))
    class_name = model.__module__ + "." + model.__class__.__name__

    # will be part of the name (in brackets)
    sub_components_names = ""
    for key in sub_components:
        if key in sub_components_explicit:
            sub_components_names += "," + key + "=" + sub_components[key].name
        else:
            sub_components_names += "," + sub_components[key].name

    if sub_components_names:
        # slice operation on string in order to get rid of leading comma
        name = '%s(%s)' % (class_name, sub_components_names[1:])
    else:
        name = class_name

    # Get the external versions of all sub-components
    external_version = _get_external_version_string(model, sub_components)

    dependencies = [_format_external_version('sklearn', sklearn.__version__),
                    'numpy>=1.6.1', 'scipy>=0.9']
    dependencies = '\n'.join(dependencies)

    flow = OpenMLFlow(name=name,
                      class_name=class_name,
                      description='Automatically created scikit-learn flow.',
                      model=model,
                      components=sub_components,
                      parameters=parameters,
                      parameters_meta_info=parameters_meta_info,
                      external_version=external_version,
                      tags=['openml-python', 'sklearn', 'scikit-learn',
                            'python',
                            _format_external_version('sklearn',
                                                     sklearn.__version__).replace('==', '_'),
                            # TODO: add more tags based on the scikit-learn
                            # module a flow is in? For example automatically
                            # annotate a class of sklearn.svm.SVC() with the
                            # tag svm?
                            ],
                      language='English',
                      # TODO fill in dependencies!
                      dependencies=dependencies)

    #global DEBUG
    if(DEBUG):
        print("{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(flow.name,flow.class_name,flow.description,flow.model,flow.components,flow.parameters,parameters_meta_info,flow.external_version,flow.tags,flow.language,flow.dependencies))
    return flow

def _serialize_sequential_model(model):
    """Create an OpenMLFlow.
    
    Calls `sklearn_to_flow` recursively to properly serialize the
    parameters to strings and the components (other models) to OpenMLFlows.

    Parameters
    ----------
    model : sklearn estimator

    Returns
    -------
    OpenMLFlow

    """
    print('seq')
    # Get all necessary information about the model objects itself
    parameters, parameters_meta_info, sub_components, sub_components_explicit =\
        _extract_information_from_seq(model)
    if(DEBUG):
        print(subcomponents)    

    #print("{},{},{},{}".format(parameters, parameters_meta_info,sub_components, sub_components_explicit))
    # Check that a component does not occur multiple times in a flow as this
    # is not supported by OpenML
    _check_multiple_occurence_of_component_in_flow(model, sub_components)

    # Create a flow name, which contains all components in brackets, for
    # example RandomizedSearchCV(Pipeline(StandardScaler,AdaBoostClassifier(DecisionTreeClassifier)),StandardScaler,AdaBoostClassifier(DecisionTreeClassifier))
    class_name = model.__module__ + "." + model.__class__.__name__

    # will be part of the name (in brackets)
    sub_components_names = ""
    for key in sub_components:
        if key in sub_components_explicit:
            sub_components_names += "," + key + "=" + sub_components[key].name
        else:
            sub_components_names += "," + sub_components[key].name

    if sub_components_names:
        # slice operation on string in order to get rid of leading comma
        name = '%s(%s)' % (class_name, sub_components_names[1:])
    else:
        name = class_name

    # Get the external versions of all sub-components
    external_version = _get_external_version_string(model, sub_components)

    dependencies = [_format_external_version('sklearn', sklearn.__version__),
                    'numpy>=1.6.1', 'scipy>=0.9']
    dependencies = '\n'.join(dependencies)


    flow = OpenMLFlow(name=name,
                      class_name=class_name,
                      description='Automatically created scikit-learn flow.',
                      model=model,
                      components=sub_components,
                      parameters=parameters,
                      parameters_meta_info=parameters_meta_info,
                      external_version=external_version,
                      tags=['openml-python', 'sklearn', 'scikit-learn',
                            'python',
                            _format_external_version('sklearn',
                                                     sklearn.__version__).replace('==', '_'),
                            # TODO: add more tags based on the scikit-learn
                            # module a flow is in? For example automatically
                            # annotate a class of sklearn.svm.SVC() with the
                            # tag svm?
                            ],
                      language='English',
                      # TODO fill in dependencies!
                      dependencies=dependencies)
    print("{},{},{},{},{},{},{},{},{},{},{}".format(flow.name,flow.class_name,flow.description,flow.model,flow.components,flow.parameters,parameters_meta_info,flow.external_version,flow.tags,flow.language,flow.dependencies))
    return flow

def _get_external_version_string(model, sub_components):
    # Create external version string for a flow, given the model and the
    # already parsed dictionary of sub_components. Retrieves the external
    # version of all subcomponents, which themselves already contain all
    # requirements for their subcomponents. The external version string is a
    # sorted concatenation of all modules which are present in this run.
    model_package_name = model.__module__.split('.')[0]
    module = importlib.import_module(model_package_name)
    model_package_version_number = module.__version__
    external_version = _format_external_version(model_package_name,
                                                model_package_version_number)
    openml_version = _format_external_version('openml', openml.__version__)
    external_versions = set()
    external_versions.add(external_version)
    external_versions.add(openml_version)
    for visitee in sub_components.values():
        for external_version in visitee.external_version.split(','):
            external_versions.add(external_version)
    external_versions = list(sorted(external_versions))
    external_version = ','.join(external_versions)
    return external_version


def _check_multiple_occurence_of_component_in_flow(model, sub_components):
    #print(model)
    #print('--------')
    #print(sub_components)
    #print('==========')
    to_visit_stack = []
    to_visit_stack.extend(sub_components.values())
    known_sub_components = set()
    while len(to_visit_stack) > 0:
        visitee = to_visit_stack.pop()
        print(visitee)
        if visitee.name in known_sub_components:
            raise ValueError('Found a second occurence of component %s when '
                             'trying to serialize %s.' % (visitee.name, model))
        else:
            known_sub_components.add(visitee.name)
            to_visit_stack.extend(visitee.components.values())


def translate_seq(model): #return list of models
    #steps=[]
    steps={}
    for layer_number,submodel in enumerate(model.get_config()[0:2]):
        layer_order="layer{}".format(layer_number)
        steps[layer_order]=[(i,j)for i,j in submodel.items()]
        #if(submodel['class_name']=='Dense'):
        #    steps.append(("layer{}".format(layer_number),Dense.from_config(submodel['config'])))
        #elif(submodel['class_name']=='Dropout'):
        #    steps.append(("layer{}".format(layer_number),Dropout.from_config(submodel['config'])))
    return steps
def _extract_information_from_model(model):
    # This function contains four "global" states and is quite long and
    # complicated. If it gets to complicated to ensure it's correctness,
    # it would be best to make it a class with the four "global" states being
    # the class attributes and the if/elif/else in the for-loop calls to
    # separate class methods

    # stores all entities that should become subcomponents
    sub_components = OrderedDict()
    # stores the keys of all subcomponents that should become
    sub_components_explicit = set()
    parameters = OrderedDict()
    parameters_meta_info = OrderedDict()

    #if(isinstance(model, keras.models.Sequential)):
    #    model_parameters = model.get_config()
    #else:
    #if(hasattr(model_parameters, 'build_fn')):
    #    model_parameters[build_fn]=model.build_fn().get_config()
    #global seq_flag
    #if(not seq_flag):
    #    model_parameters[model]=model.build_fn().get_config()
    #    seq_flag=True

    model_parameters = model.get_params(deep=True)
    #print("=================debug line start")
    #print("model params=".format(model_parameters))
    #print(model_parameters)
    #print("=================debug line end")
    for k, v in model_parameters.items():#sorted(model_parameters.items(), key=lambda t: t[0]):
        rval = sklearn_to_flow(v, model)
        #print('rval={}, k={}, v={}'.format(rval,k,v))
        

        if (isinstance(rval, (list, tuple)) and len(rval) > 0 and
                isinstance(rval[0], (list, tuple)) and
                [type(rval[0]) == type(rval[i]) for i in range(len(rval))]):

            # Steps in a pipeline or feature union, or base classifiers in voting classifier
            parameter_value = list()
            reserved_keywords = set(model.get_params(deep=False).keys())

            for sub_component_tuple in rval:
                identifier, sub_component = sub_component_tuple
                sub_component_type = type(sub_component_tuple)

                if identifier in reserved_keywords:
                    parent_model_name = model.__module__ + "." + \
                                        model.__class__.__name__
                    raise PyOpenMLError('Found element shadowing official ' + \
                                        'parameter for %s: %s' % (parent_model_name, identifier))

                if sub_component is None:
                    # In a FeatureUnion it is legal to have a None step

                    pv = [identifier, None]
                    if sub_component_type is tuple:
                        pv = tuple(pv)
                    parameter_value.append(pv)

                else:
                    # Add the component to the list of components, add a
                    # component reference as a placeholder to the list of
                    # parameters, which will be replaced by the real component
                    # when deserializing the parameter
                    sub_components_explicit.add(identifier)
                    sub_components[identifier] = sub_component
                    component_reference = OrderedDict()
                    component_reference[
                        'oml-python:serialized_object'] = 'component_reference'
                    cr_value = OrderedDict()
                    cr_value['key'] = identifier
                    cr_value['step_name'] = identifier
                    component_reference['value'] = cr_value
                    parameter_value.append(component_reference)

            if isinstance(rval, tuple):
                parameter_value = tuple(parameter_value)

            # Here (and in the elif and else branch below) are the only
            # places where we encode a value as json to make sure that all
            # parameter values still have the same type after
            # deserialization
            parameter_value = json.dumps(parameter_value)
            parameters[k] = parameter_value

        elif isinstance(rval, OpenMLFlow):
            # A subcomponent, for example the base model in
            # AdaBoostClassifier
            sub_components[k] = rval
            sub_components_explicit.add(k)
            component_reference = OrderedDict()
            component_reference[
                'oml-python:serialized_object'] = 'component_reference'
            cr_value = OrderedDict()
            cr_value['key'] = k
            cr_value['step_name'] = None
            component_reference['value'] = cr_value
            component_reference = sklearn_to_flow(component_reference, model)
            parameters[k] = json.dumps(component_reference)

        else:

            # a regular hyperparameter
            if not (hasattr(rval, '__len__') and len(rval) == 0):
                rval = json.dumps(rval)
                parameters[k] = rval
            else:
                parameters[k] = None

        parameters_meta_info[k] = OrderedDict((('description', None),
                                               ('data_type', None)))

    #print("output=\n{}\n\n {}\n\n {}\n\n {}\n\n".format(parameters, parameters_meta_info, sub_components, sub_components_explicit))

    return parameters, parameters_meta_info, sub_components, sub_components_explicit


def _extract_information_from_seq(model):
    # This function contains four "global" states and is quite long and
    # complicated. If it gets to complicated to ensure it's correctness,
    # it would be best to make it a class with the four "global" states being
    # the class attributes and the if/elif/else in the for-loop calls to
    # separate class methods

    # stores all entities that should become subcomponents
    sub_components = OrderedDict()
    # stores the keys of all subcomponents that should become
    sub_components_explicit = set()
    parameters = OrderedDict()
    parameters_meta_info = OrderedDict()
    for i in range(len(model.get_config())):
        model_parameters = model.get_config()
        #print("model params=".format(model_parameters))
        #print(model_parameters)
    
        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = sklearn_to_flow(v, model)
            #print('rval={}, k={}, v={}'.format(rval,k,v))

            #if(isinstance(rval,dict)):
            #   rval=[rval]#[(k, v) for k, v in rval.items()]
            #print('rval={}, i={}'.format(rval,i))
            if (isinstance(rval, (list, tuple)) and len(rval) > 0 and
                    isinstance(rval[0], (list, tuple)) and
                    [type(rval[0]) == type(rval[i]) for i in range(len(rval))]):
                # Steps in a pipeline or feature union, or base classifiers in voting classifier
                parameter_value = list()
                reserved_keywords = set(model.get_config().keys())

                for sub_component_tuple in rval:
                    identifier, sub_component = sub_component_tuple
                    sub_component_type = type(sub_component_tuple)

                    if identifier in reserved_keywords:
                        parent_model_name = model.__module__ + "." + \
                                            model.__class__.__name__
                        raise PyOpenMLError('Found element shadowing official ' + \
                                            'parameter for %s: %s' % (parent_model_name, identifier))

                    if sub_component is None:
                        # In a FeatureUnion it is legal to have a None step

                        pv = [identifier, None]
                        if sub_component_type is tuple:
                            pv = tuple(pv)
                        parameter_value.append(pv)

                    else:
                    # Add the component to the list of components, add a
                    # component reference as a placeholder to the list of
                    # parameters, which will be replaced by the real component
                    # when deserializing the parameter
                        sub_components_explicit.add(identifier)
                        sub_components[identifier] = sub_component
                        component_reference = OrderedDict()
                        component_reference[
                            'oml-python:serialized_object'] = 'component_reference'
                        cr_value = OrderedDict()
                        cr_value['key'] = identifier
                        cr_value['step_name'] = identifier
                        component_reference['value'] = cr_value
                        parameter_value.append(component_reference)

                if isinstance(rval, tuple):
                    parameter_value = tuple(parameter_value)

                # Here (and in the elif and else branch below) are the only
                # places where we encode a value as json to make sure that all
                # parameter values still have the same type after
                # deserialization
                parameter_value = json.dumps(parameter_value)
                parameters[k] = parameter_value
            
            elif isinstance(rval, OpenMLFlow):

                # A subcomponent, for example the base model in
                # AdaBoostClassifier
                sub_components[k] = rval
                sub_components_explicit.add(k)
                component_reference = OrderedDict()
                component_reference[
                    'oml-python:serialized_object'] = 'component_reference'
                cr_value = OrderedDict()
                cr_value['key'] = k
                cr_value['step_name'] = None
                component_reference['value'] = cr_value
                component_reference = sklearn_to_flow(component_reference, model)
                parameters[k] = json.dumps(component_reference)

            else:

                # a regular hyperparameter
                if not (hasattr(rval, '__len__') and len(rval) == 0):
                    rval = json.dumps(rval)
                    parameters[k] = rval
                else:
                    parameters[k] = None

            parameters_meta_info[k] = OrderedDict((('description', None),
                                               ('data_type', None)))
    #print("output={}\n\n {}\n\n {}\n\n {}\n\n".format(parameters, parameters_meta_info, sub_components, sub_components_explicit))
    return parameters, parameters_meta_info, sub_components, sub_components_explicit


def _deserialize_model(flow, **kwargs):

    model_name = flow.class_name
    _check_dependencies(flow.dependencies)

    parameters = flow.parameters
    components = flow.components
    parameter_dict = OrderedDict()

    # Do a shallow copy of the components dictionary so we can remove the
    # components from this copy once we added them into the pipeline. This
    # allows us to not consider them any more when looping over the
    # components, but keeping the dictionary of components untouched in the
    # original components dictionary.
    components_ = copy.copy(components)

    for name in parameters:
        value = parameters.get(name)
        rval = flow_to_sklearn(value, components=components_)
        parameter_dict[name] = rval

    for name in components:
        if name in parameter_dict:
            continue
        if name not in components_:
            continue
        value = components[name]
        rval = flow_to_sklearn(value)
        parameter_dict[name] = rval

    module_name = model_name.rsplit('.', 1)
    try:
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
    except:
        warnings.warn('Cannot create model %s for flow.' % model_name)
        return None

    return model_class(**parameter_dict)

def _deserialize_seq_model(flow, **kwargs):

    model_name = flow.class_name
    _check_dependencies(flow.dependencies)

    parameters = flow.parameters
    components = flow.components
    parameter_dict = OrderedDict()

    # Do a shallow copy of the components dictionary so we can remove the
    # components from this copy once we added them into the pipeline. This
    # allows us to not consider them any more when looping over the
    # components, but keeping the dictionary of components untouched in the
    # original components dictionary.
    components_ = copy.copy(components)

    for name in parameters:
        value = parameters.get(name)
        rval = flow_to_sklearn(value, components=components_)
        parameter_dict[name] = rval

    for name in components:
        if name in parameter_dict:
            continue
        if name not in components_:
            continue
        value = components[name]
        rval = flow_to_sklearn(value)
        parameter_dict[name] = rval

    module_name = model_name.rsplit('.', 1)
    try:
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
    except:
        warnings.warn('Cannot create model %s for flow.' % model_name)
        return None

    return model_class(**parameter_dict)



def _check_dependencies(dependencies):
    if not dependencies:
        return

    dependencies = dependencies.split('\n')
    for dependency_string in dependencies:
        match = DEPENDENCIES_PATTERN.match(dependency_string)
        dependency_name = match.group('name')
        operation = match.group('operation')
        version = match.group('version')

        module = importlib.import_module(dependency_name)
        required_version = LooseVersion(version)
        installed_version = LooseVersion(module.__version__)

        if operation == '==':
            check = required_version == installed_version
        elif operation == '>':
            check = installed_version > required_version
        elif operation == '>=':
            check = installed_version > required_version or \
                    installed_version == required_version
        else:
            raise NotImplementedError(
                'operation \'%s\' is not supported' % operation)
        if not check:
            raise ValueError('Trying to deserialize a model with dependency '
                             '%s not satisfied.' % dependency_string)


def serialize_type(o):
    mapping = {float: 'float',
               np.float: 'np.float',
               np.float32: 'np.float32',
               np.float64: 'np.float64',
               int: 'int',
               np.int: 'np.int',
               np.int32: 'np.int32',
               np.int64: 'np.int64'}
    ret = OrderedDict()
    ret['oml-python:serialized_object'] = 'type'
    ret['value'] = mapping[o]
    return ret

def deserialize_type(o, **kwargs):
    mapping = {'float': float,
               'np.float': np.float,
               'np.float32': np.float32,
               'np.float64': np.float64,
               'int': int,
               'np.int': np.int,
               'np.int32': np.int32,
               'np.int64': np.int64}
    return mapping[o]


def serialize_rv_frozen(o):
    args = o.args
    kwds = o.kwds
    a = o.a
    b = o.b
    dist = o.dist.__class__.__module__ + '.' + o.dist.__class__.__name__
    ret = OrderedDict()
    ret['oml-python:serialized_object'] = 'rv_frozen'
    ret['value'] = OrderedDict((('dist', dist), ('a', a), ('b', b),
                                ('args', args), ('kwds', kwds)))
    return ret

def deserialize_rv_frozen(o, **kwargs):
    args = o['args']
    kwds = o['kwds']
    a = o['a']
    b = o['b']
    dist_name = o['dist']

    module_name = dist_name.rsplit('.', 1)
    try:
        rv_class = getattr(importlib.import_module(module_name[0]),
                           module_name[1])
    except:
        warnings.warn('Cannot create model %s for flow.' % dist_name)
        return None

    dist = scipy.stats.distributions.rv_frozen(rv_class(), *args, **kwds)
    dist.a = a
    dist.b = b

    return dist

def serialize_sequence(o):
    ##TODO Irfan
    ret = OrderedDict()
    ret['oml-python:serialized_object'] = 'sequence'
    ret['value'] = o
    return ret

def deserialize_sequence(o):
    ##TODO Irfan
    build_fn = o['build_fn']
    verbose = o['verbose']
    epoch = o['epoch']
    bacth_size = o['batch_size']
    
    module_name = dist_name.rsplit('.', 1)
    try:
        rv_class = getattr(importlib.import_module(module_name[0]),
                           module_name[1])
    except:
        warnings.warn('Cannot create model %s for flow.' % dist_name)
        return None

    dist = scipy.stats.distributions.rv_frozen(rv_class(), *args, **kwds)
    dist.a = a
    dist.b = b

def serialize_function(o):
    name = o.__module__ + '.' + o.__name__
    ret = OrderedDict()
    ret['oml-python:serialized_object'] = 'function'
    #model_copy = keras.wrappers.scikit_learn.KerasClassifier(build_fn=o)
    ret['value'] = o().get_config()
    
    #a=keras.models.Sequential()
    #a.add(keras.layers.Reshape((28, 28, 1), input_shape=(784,)))
    #if(isinstance(a,keras.models.Sequential)):
    #    ret['value'] = a.get_config()
    #else:
    #    ret['value'] = name
    #return _serialize_sequential_model(a)
   
    
    return ret


def deserialize_function(name, **kwargs):
    module_name = name.rsplit('.', 1)
    try:
        function_handle = getattr(importlib.import_module(module_name[0]),
                                  module_name[1])
    except Exception as e:
        warnings.warn('Cannot load function %s due to %s.' % (name, e))
        return None
    return function_handle

def _serialize_cross_validator(o):
    ret = OrderedDict()

    parameters = OrderedDict()

    # XXX this is copied from sklearn.model_selection._split
    cls = o.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])

    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(o, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)

        if not (hasattr(value, '__len__') and len(value) == 0):
            value = json.dumps(value)
            parameters[key] = value
        else:
            parameters[key] = None

    ret['oml-python:serialized_object'] = 'cv_object'
    name = o.__module__ + "." + o.__class__.__name__
    value = OrderedDict([['name', name], ['parameters', parameters]])
    ret['value'] = value

    return ret

def _check_n_jobs(model):
    '''
    Returns True if the parameter settings of model are chosen s.t. the model
     will run on a single core (in that case, openml-python can measure runtimes)
    '''
    def check(param_dict, disallow_parameter=False):
        for param, value in param_dict.items():
            # n_jobs is scikitlearn parameter for paralizing jobs
            if param.split('__')[-1] == 'n_jobs':
                # 0 = illegal value (?), 1 = use one core,  n = use n cores
                # -1 = use all available cores -> this makes it hard to
                # measure runtime in a fair way
                if value != 1 or disallow_parameter:
                    return False
        return True

    if not (isinstance(model, sklearn.base.BaseEstimator) or
            isinstance(model, sklearn.model_selection._search.BaseSearchCV) or
            isinstance(model, keras.wrappers.scikit_learn.KerasClassifier)):
        raise ValueError('model should be BaseEstimator or BaseSearchCV')

    # make sure that n_jobs is not in the parameter grid of optimization procedure
    if isinstance(model, sklearn.model_selection._search.BaseSearchCV):
        param_distributions = None
        if isinstance(model, sklearn.model_selection.GridSearchCV):
            param_distributions = model.param_grid
        elif isinstance(model, sklearn.model_selection.RandomizedSearchCV):
            param_distributions = model.param_distributions
        else:
            if hasattr(model, 'param_distributions'):
                param_distributions = model.param_distributions
            else:
                raise AttributeError('Using subclass BaseSearchCV other than {GridSearchCV, RandomizedSearchCV}. Could not find attribute param_distributions. ')
            print('Warning! Using subclass BaseSearchCV other than ' \
                  '{GridSearchCV, RandomizedSearchCV}. Should implement param check. ')
            
        if not check(param_distributions, True):
            raise PyOpenMLError('openml-python should not be used to '
                                'optimize the n_jobs parameter.')

    # check the parameters for n_jobs
    return check(model.get_params(), False)

def _deserialize_cross_validator(value, **kwargs):
    model_name = value['name']
    parameters = value['parameters']

    module_name = model_name.rsplit('.', 1)
    model_class = getattr(importlib.import_module(module_name[0]),
                          module_name[1])
    for parameter in parameters:
        parameters[parameter] = flow_to_sklearn(parameters[parameter])
    return model_class(**parameters)


def _format_external_version(model_package_name, model_package_version_number):
    return '%s==%s' % (model_package_name, model_package_version_number)
