"""
This is a boilerplate pipeline 'nome_modelo'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (importar_dados,data_metrics,preparar_dados,model_lr,report_model,model_lda)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=importar_dados,
            inputs=['params:shot'],
            outputs= 'filtered_df',
            name='dados_importados',
        ),

        node(
            func=data_metrics,
            inputs='filtered_df',
            outputs= 'filtered_df_metrics',
        ),

        node(
            func=preparar_dados,
            inputs=['filtered_df','params:test_size','params:random_seed'],
            outputs= ['X_train', 'X_test', 'y_train', 'y_test'],
            name='datasets',
        ),

        node(
            func=model_lr,
            inputs=['X_train', 'X_test', 'y_train', 'y_test'],
            outputs= 'lr_model',
            name='model_lr',
        ),

        node(
            func=report_model,
            inputs='lr_model',
            outputs= 'metrics_dict',
            name='report_model',    
        ),

        node(
            func=model_lda,
            inputs=['params:shot','params:test_size','params:random_seed'],
            outputs= 'lda_model',
            name='model_lda',
        ),

    ])
