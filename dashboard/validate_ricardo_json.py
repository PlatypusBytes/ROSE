from schema import Schema, And, SchemaError


def definitions():
    """
    Defines the configuration schema for the validation of the input json file

    @return: Schema configuration file
    """

    # sos data dictionary
    conf_schema_ricardo = Schema({
        "coordinates": And([[float], [float]]),
        "speed": And([float]),
        "axle_acc": And([float])
    })

    # main scheme
    conf_schema = Schema({"project_name": str,
                          "data": {And(str): conf_schema_ricardo}})

    return conf_schema


def check_json(input_json):
    """
    Check if the input json file is correct

    @param input_json: input json file
    @return: True / False
    """
    conf_schema = definitions()
    try:
        conf_schema.validate(input_json)
        return True
    except SchemaError:
        return False
