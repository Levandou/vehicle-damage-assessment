"""Run the two-view research pipeline with locally supplied model files."""
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description='Analyse far and near photographs of one vehicle.')
    parser.add_argument('--far', required=True, help='Path to the wider vehicle photograph')
    parser.add_argument('--near', required=True, help='Path to the close-up photograph')
    parser.add_argument('--models', default='some_models', help='Directory containing model files')
    parser.add_argument('--output', default='results', help='Directory for generated results')
    args = parser.parse_args()

    models = Path(args.models)
    expected = {
        'details': models / 'details_deeplabv3plus.pt',
        'damages': models / 'damages_maskrcnn.pth',
        'points': models / 'points_hrnet_w32.pt',
        'geometry': models / 'hyundai_solaris.glb',
    }
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        parser.error('Missing model files: ' + ', '.join(missing))

    from damage_pipeline import DamageEstimationPipeline
    from near_pipeline import run_two_view

    hidden = models / 'xgboost_hidden_damage_model.pkl'
    pipeline = DamageEstimationPipeline(
        details_model_path=str(expected['details']),
        damages_model_path=str(expected['damages']),
        points_model_path=str(expected['points']),
        glb_path=str(expected['geometry']),
        hidden_damage_model_path=str(hidden) if hidden.is_file() else None,
    )
    report = run_two_view(args.near, args.far, args.output, pipeline)
    print('Status:', report.get('status', 'unknown'))
    print('Results:', Path(args.output).resolve())


if __name__ == '__main__':
    main()
