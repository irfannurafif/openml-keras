Hi, I am trying to add Keras integration to OpenML. But, when I try to run this code:

  ```python
  from keras.wrappers.scikit_learn import KerasClassifier
   def create_model():
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(784,)))

    # first set of CONV => RELU => POOL
    model.add(Conv2D(20, (5, 5), padding="same",
    input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    # softmax classifier
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    return model

m=KerasClassifier(build_fn=create_model, epochs=1, batch_size=10,verbose=1)
run = tes_oml.runs.run_model_on_task(task, m)
myrun = run.publish()
print("Uploaded to http://www.openml.org/r/" + str(myrun.run_id))
```
I got an OpenMLServerException: Could not validate run xml by xsd.

```python
---------------------------------------------------------------------------
OpenMLServerException                     Traceback (most recent call last)
<ipython-input-108-b2327e2ed075> in <module>()
      1 #task = oml.tasks.get_task(3573)
----> 2 run2 = tes_oml.runs.run_model_on_task(task, m)
      3 #run = tes_oml.runs.run_model_on_task(task, model2)
      4 myrun2 = run2.publish()
      5 print("Uploaded to http://www.openml.org/r/" + str(myrun2.run_id))

D:\irfan\kuliah\TUE\Q7\thesis\openml-python\openml\runs\functions.py in run_model_on_task(task, model, avoid_duplicate_runs, flow_tags, seed)
     88     return run_flow_on_task(task=task, flow=flow,
     89                             avoid_duplicate_runs=avoid_duplicate_runs,
---> 90                             flow_tags=flow_tags, seed=seed)
     91 
     92 

D:\irfan\kuliah\TUE\Q7\thesis\openml-python\openml\runs\functions.py in run_flow_on_task(task, flow, avoid_duplicate_runs, flow_tags, seed)
    132     if avoid_duplicate_runs and flow_id:
    133         flow_from_server = get_flow(flow_id)
--> 134         setup_id = setup_exists(flow_from_server, flow.model)
    135         ids = _run_exists(task.task_id, setup_id)
    136         if ids:

D:\irfan\kuliah\TUE\Q7\thesis\openml-python\openml\setups\functions.py in setup_exists(flow, model)
     46 
     47     result = openml._api_calls._perform_api_call('/setup/exists/',
---> 48                                                  file_elements=file_elements)
     49     result_dict = xmltodict.parse(result)
     50     setup_id = int(result_dict['oml:setup_exists']['oml:id'])

D:\irfan\kuliah\TUE\Q7\thesis\openml-python\openml\_api_calls.py in _perform_api_call(call, data, file_dictionary, file_elements, add_authentication)
     50     if file_dictionary is not None or file_elements is not None:
     51         return _read_url_files(url, data=data, file_dictionary=file_dictionary,
---> 52                                file_elements=file_elements)
     53     return _read_url(url, data)
     54 

D:\irfan\kuliah\TUE\Q7\thesis\openml-python\openml\_api_calls.py in _read_url_files(url, data, file_dictionary, file_elements)
     96     response = requests.post(url, data=data, files=file_elements)
     97     if response.status_code != 200:
---> 98         raise _parse_server_exception(response, url=url)
     99     if 'Content-Encoding' not in response.headers or \
    100             response.headers['Content-Encoding'] != 'gzip':

OpenMLServerException: Could not validate run xml by xsd.
```
