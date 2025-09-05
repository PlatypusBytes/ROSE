# ROSE Software Architecture

```mermaid
graph TD
    %% Main components
    User[User Interface / Scripts]
    PreProcess[Pre-Processing]
    Model[Model Components]
    Solvers[Solvers]
    Interaction[Train-Track Interaction]
    PostProcess[Post-Processing]

    %% Sub-components
    subgraph "Model Components"
        ModelPart[Model Parts]
        Geometry[Geometry]
        TrackModel[Track Model]
        TrainModel[Train Model]
        SoilModel[Soil Model]
    end

    subgraph "Component Details"
        %% Track components
        Rail[Rail]
        RailPad[Rail Pad]
        Sleeper[Sleeper]
        Soil[Soil]

        %% Train components
        Carts[Carts]
        Bogies[Bogies]
        Wheels[Wheels]

        %% Material properties
        Material[Material Properties]
        Section[Section Properties]
    end

    %% Main workflow
    User --> PreProcess
    PreProcess --> Model
    Model --> Interaction
    Interaction --> Solvers
    Solvers --> PostProcess

    %% Model connections
    ModelPart --- TrackModel
    ModelPart --- TrainModel
    ModelPart --- SoilModel
    Geometry --- TrackModel
    Geometry --- TrainModel

    %% Component connections
    TrackModel --- Rail
    TrackModel --- RailPad
    TrackModel --- Sleeper
    TrackModel --- Soil
    TrainModel --- Carts
    Carts --- Bogies
    Bogies --- Wheels

    %% Properties
    Material --- Rail
    Material --- RailPad
    Material --- Sleeper
    Material --- Soil
    Section --- Rail

    %% Legend
    classDef core fill:#f9f,stroke:#333,stroke-width:2px
    classDef component fill:#bbf,stroke:#333,stroke-width:1px
    classDef interface fill:#bfb,stroke:#333,stroke-width:1px

    class User,PreProcess,Model,Solvers,Interaction,PostProcess core
    class ModelPart,Geometry,TrackModel,TrainModel,SoilModel component
    class Rail,RailPad,Sleeper,Soil,Carts,Bogies,Wheels,Material,Section interface
```

## Description

The ROSE software architecture consists of several key components:

1. **Pre-Processing**:
   - Handles input data preparation
   - Geometry creation and mesh generation
   - Material property definition
   - Train configurations

2. **Model Components**:
   - **Track Model**: Rail, rail pad, sleeper, and soil elements
   - **Train Model**: Carts, bogies, and wheels
   - **Soil Model**: Soil layers with different properties

3. **Train-Track Interaction**:
   - Manages the coupling between train and track
   - Handles contact mechanics (Hertzian contact)
   - Applies train loading to track

4. **Solvers**:
   - Numerical methods for time integration
   - Includes Newmark, HHT, and other solvers
   - Handles dynamic analysis

5. **Post-Processing**:
   - Result collection and storage
   - Visualization tools
   - Settlement calculations
   - Long-term performance analysis

This architecture enables simulation of train-track dynamic interaction and prediction of long-term track degradation through sequential application of each component.