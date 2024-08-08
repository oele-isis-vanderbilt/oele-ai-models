# oele-ai-models
A repository for inference endpoints of various models

## Local Deployment
After cloning the repository

### Serving The Model

We're now ready to serve our model with MLServer. To do that we can simply run:

```bash
mlserver start hsemotion/
```

MLServer will now start up, load our cassava model and provide access through both a REST and gRPC API.

### Making Predictions Using The API

Now that our API is up and running. Open a new terminal window and navigate back to the root of this repository. We can then send predictions to our api using the `test.py` file by running:

```bash
python exmaples/hsemotion/test.py --local
```