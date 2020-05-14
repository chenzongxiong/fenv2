# SETTINGS
import os

DEBUG_INIT_TF_VALUE = False
WEIGHTS = [1]
WIDTHS = [1]
METHODS = ["sin"]
# UNITS = [4, 8, 16]
UNITS = [1, 8, 20, 100]
# NB_PLAYS = [1, 4, 10, 20]
NB_PLAYS = [1, 20 , 40, 100]

EPOCHS = 20000
POINTS = 500
# NOTE: trick here, batch_size must be always equal to 1
BATCH_SIZE = 1

BATCH_SIZE_LIST = [10]

FNAME_FORMAT = dict(
    operators="./training-data/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/base.csv",
    operators_predictions="./training-data/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/predictions-{loss}.csv",
    operators_loss="./training-data/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/loss-{loss}.csv",
    operators_gif="./pics/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/base-{loss}.gif",
    operators_gif_snake="./pics/operators/method-{method}/weight-{weight}/width-{width}/points-{points}/snake-{loss}.gif",

    plays="./training-data/plays/method-{method}/weight-{weight}/width-{width}/points-{points}/base.csv",
    plays_predictions="./training-data/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/predictions-{loss}.csv",
    plays_loss = "./training-data/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/loss-{loss}.csv",
    plays_gif="./pics/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/base-{loss}.gif",
    plays_gif_snake="./pics/plays/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/points-{points}/snake-{loss}.gif",

    models="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/points-{points}/base.csv",
    models_loss="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/batch_size-{batch_size}/points-{points}/loss-{loss}.csv",
    models_loss_history="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/batch_size-{batch_size}/points-{points}/history-{loss}.csv",
    models_predictions="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/batch_size-{batch_size}/points-{points}/predictions-{loss}.csv",
    models_gif="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/points-{points}/base-{loss}.gif",
    models_gif_snake="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/points-{points}/snake-{loss}.gif",

    models_rnn_loss="./training-data/models/rnn/{method}-{weight}-{width}-{nb_plays}-rnn-loss.csv",
    models_rnn_predictions="./training-data/models/rnn/{method}-{weight}-{width}-{nb_plays}-rnn-predictions.csv",
    models_lstm_loss="./training-data/models/lstm/{method}-{weight}-{width}-{nb_plays}-lstm-loss.csv",
    models_lstm_predictions="./training-data/models/lstm/{method}-{weight}-{width}-{nb_plays}-lstm-predictions.csv",
    models_gru_loss="./training-data/models/gru/{method}-{weight}-{width}-{nb_plays}-gru-loss.csv",
    models_gru_predictions="./training-data/models/gru/{method}-{weight}-{width}-{nb_plays}-gru-predictions.csv",

    # G model
    models_G="./training-data/G/{method}-{weight}-{width}-{nb_plays}.csv",
    # models_G_multi="./training-data/G/{method}-{weight}-{width}-{nb_plays}-multi.csv",
    models_G_loss="./training-data/G/{method}-{weight}-{width}-{nb_plays}-{batch_size}-loss.csv",
    models_G_predictions="./training-data/G/{method}-{weight}-{width}-{nb_plays}-{batch_size}-predictions.csv",
    # F model
    models_F="./training-data/F/{method}-{weight}-{width}-{nb_plays}.csv",
    # models_F_multi="./training-data/F/{method}-{weight}-{width}-{nb_plays}-multi.csv",
    models_F_loss="./training-data/F/{method}-{weight}-{width}-{nb_plays}-{batch_size}-loss.csv",
    models_F_predictions="./training-data/F/{method}-{weight}-{width}-{nb_plays}-{batch_size}-predictions.csv",
    models_F_gif="./pics/F/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}.gif",
    models_F_gif_snake="./pics/F/{method}-{weight}-{width}-{nb_plays}-{nb_plays_}-{batch_size}-snake.gif",


    # operator noise
    operators_noise="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/base.csv",
    operators_noise_predictions="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/predictions-{loss}.csv",
    operators_noise_loss="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/loss-{loss}.csv",
    operators_noise_loss_histroy="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/loss-history-{loss}.csv",
    operators_noise_gif="./pics/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/base-{loss}.gif",
    operators_noise_gif_snake="./pics/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/snake-{loss}.gif",

    # play noise
    plays_noise="./training-data/plays/{method}-{weight}-{width}-{mu}-{sigma}-tanh.csv",
    plays_noise_predictions="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-predictions.csv",
    plays_noise_loss="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-loss.csv",
    plays_noise_loss_history="./training-data/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-loss-history.csv",
    plays_noise_gif="./pics/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}.gif",
    plays_noise_gif_snake="./pics/plays/{method}-{weight}-{width}-{activation}-{units}-{mu}-{sigma}-snake.gif",

    # model noise
    models_noise="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base.csv",
    models_noise_predictions="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/predictions-{loss}.csv",
    models_noise_loss_history="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-history-{loss}.csv",
    models_noise_loss="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-{loss}.csv",
    models_noise_gif="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/base-{loss}.gif",
    models_noise_gif_snake="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/snake-{loss}.gif",
    models_noise_ts_outputs_gif="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/ts-outputs-{loss}.gif",
    models_noise_saved_weights="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/weights-{loss}.h5",


    models_nb_plays_noise="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base.csv",
    models_nb_plays_noise_predictions="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/predictions-{loss}.csv",
    models_nb_plays_noise_loss_history="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-history-{loss}.csv",
    models_nb_plays_noise_loss="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-{loss}.csv",
    models_nb_plays_noise_gif="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/base-{loss}.gif",
    models_nb_plays_noise_gif_snake="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/snake-{loss}.gif",
    models_nb_plays_noise_ts_outputs_gif="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/ts-outputs-{loss}.gif",
    models_nb_plays_noise_saved_weights="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/weights-{loss}.h5",

    models_nb_plays_noise_interp="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/base.csv",
    models_nb_plays_noise_interp_predictions="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/predictions-{loss}.csv",
    models_nb_plays_noise_interp_loss_history="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-history-{loss}.csv",
    models_nb_plays_noise_interp_loss="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-{loss}.csv",
    models_nb_plays_noise_interp_gif="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/base-{loss}.gif",
    models_nb_plays_noise_interp_gif_snake="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/snake-{loss}.gif",
    models_nb_plays_noise_interp_ts_outputs_gif="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/ts-outputs-{loss}.gif",
    models_nb_plays_noise_interp_saved_weights="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/interp-{interp}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/weights-{loss}.h5",


    # operator test
    operators_noise_test="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/base.csv",
    operators_noise_test_predictions="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/predictions-{loss}.csv",
    operators_noise_test_loss="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/loss-{loss}.csv",
    operators_noise_test_loss_histroy="./training-data/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/state-{state}/loss-history-{loss}.csv",
    operators_noise_test_gif="./pics/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/base-{loss}.gif",
    operators_noise_test_gif_snake="./pics/operators/method-{method}/weight-{weight}/width-{width}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/snake-{loss}.gif",

    # model noise test
    models_noise_test="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/base.csv",
    models_noise_test_predictions="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/predictions-{loss}.csv",
    # models_noise_test_loss_history="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-history-{loss}.csv",
    models_noise_test_loss="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-{loss}.csv",
    models_noise_test_gif="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/base-{loss}.gif",
    models_noise_test_gif_snake="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/snake-{loss}.gif",
    models_noise_test_ts_outputs_gif="./pics/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/ts-outputs-{loss}.gif",
    models_noise_test_saved_weights="./training-data/models/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/weights-{loss}.h5",


    # model noise test
    models_nb_plays_noise_test="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/base.csv",
    models_nb_plays_noise_test_predictions="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/predictions-{loss}.csv",
    # models_nb_plays_noise_test_loss_history="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-history-{loss}.csv",
    models_nb_plays_noise_test_loss="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/loss-{loss}.csv",
    models_nb_plays_noise_test_gif="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/base-{loss}.gif",
    models_nb_plays_noise_test_gif_snake="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/snake-{loss}.gif",
    models_nb_plays_noise_test_ts_outputs_gif="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/ts-outputs-{loss}.gif",
    models_nb_plays_noise_test_saved_weights="./training-data/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/weights-{loss}.h5",


    mc="./training-data/mc/points-{points}/mu-{mu}/sigma-{sigma}/base.csv",
    # F="./training-data/F/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-{loss}.csv",
    # F_loss_history="./training-data/F/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-loss-history-{loss}.csv",
    # F_predictions="./training-data/F/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/predictions-{loss}.csv",
    # F_gif="./pics/F/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-{loss}.gif",
    # F_gif_snake="./pics/F/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/snake-{loss}.gif",
    # G_predictions="./training-data/G/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/predictions-{loss}.csv",
    # G_loss="./training-data/G/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-loss-{loss}.csv",
    # G_loss_history="./training-data/G/method-{method}/weight-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-loss-history-{loss}.csv",
    # G_gif="./pics/G/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/base-{loss}.gif",
    # G_gif_snake="./pics/F/method-{method}/weigth-{weight}/width-{width}/activation-{activation}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/snake-{loss}.gif",

    F="./training-data/F/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/base-{loss}.csv",
    F_predictions="./training-data/F/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/predictions-{loss}.csv",
    F_loss_history="./training-data/F/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/loss_history-{loss}.csv",
    F_saved_weights="./training-data/F/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/weights-{loss}.h5",
    F_gif="./pics/F/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/base-{loss}.gif",
    F_gif_snake="./pics/F/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/snake-{loss}.gif",


    F_interp="./training-data/F/interp/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/base.csv",
    F_interp_predictions="./training-data/F/interp/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/predictions-{loss}.csv",
    F_interp_loss_history="./training-data/F/interp/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/loss_history-{loss}.csv",
    F_interp_saved_weights="./training-data/F/interp/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/weights-{loss}.h5",
    F_interp_gif="./pics/F/interp/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/base-{loss}.gif",
    F_interp_gif_snake="./pics/F/interp/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/snake-{loss}.gif",

    # models_nb_plays_noise_test_ts_outputs_gif="./pics/models/diff_weights/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/state-{state}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/ts-outputs-{loss}.gif",

    F_interp_ts_outputs_gif="./pics/F/interp/method-{method}/weight-{weight}/width-{width}/nb_plays-{nb_plays}/units-{units}/mu-{mu}/sigma-{sigma}/points-{points}/nb_plays_-{nb_plays_}/batch_size-{batch_size}/state-{state}/ts-outputs-{loss}.gif",


    )

# CPU_COUNTS = os.cpu_count()
CPU_COUNTS = 40


class NetworkType:
    OPERATOR = 1
    PLAY = 2


LOG_DIR = "./log"

_prefix = './new-dataset'

DATASET_PATH = dict(
    models=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base.csv',
    models_predictions=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/predictions.csv',
    models_loss_history=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/history.csv',
    models_saved_weights=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/weights.h5',

    # Invertion
    models_invert_predictions=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/invert_predictions.csv',
    models_invert_loss_history=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/invert_history.csv',
    models_invert_saved_weights=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/invert-weights.h5',


    models_gif=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/base.gif',
    models_snake_gif=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/snake.gif',
    models_ts_outputs_gif=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ts_outputs.gif',

    models_diff_weights=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base.csv',

    # models_diff_weights_test=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/test.csv',

    models_diff_weights_predictions=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ensemble-{ensemble}/predictions.csv',

    # models_diff_weights_test_predictions=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test_predictions.csv',

    models_diff_weights_loss_history=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ensemble-{ensemble}/history.csv',
    models_diff_weights_saved_weights=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ensemble-{ensemble}/weights.h5',

    # models_diff_weights_gif=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/base.gif',
    # models_diff_weights_snake_gif=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/snake.gif',
    # models_diff_weights_ts_outputs_gif=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ts_outputs.gif',

    # models_diff_weights_test_gif=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test.gif',

    # models_diff_weights_test_snake_gif=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test_snake.gif',
    # models_diff_weights_test_ts_outputs_gif=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test_ts_outputs.gif',

    models_interp=_prefix+'/models/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base.csv',

    models_interp_gif=_prefix+'/models/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/base.gif',

    models_snake_interp_gif=_prefix+'/models/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/snake.gif',

    models_ts_outputs_interp_gif=_prefix+'/models/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ts_outputs.gif',

    models_diff_weights_interp=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base.csv',

    models_diff_weights_test_interp=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/test.csv',


    models_diff_weights_predictions_interp=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/predictions.csv',

    models_diff_weights_test_predictions_interp=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test_predictions.csv',

    models_diff_weights_loss_history_interp=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/history.csv',

    models_diff_weights_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/base.gif',

    models_diff_weights_test_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test.gif',

    models_diff_weights_snake_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/snake.gif',

    models_diff_weights_test_snake_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test_snake.gif',

    models_diff_weights_ts_outputs_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ts_outputs.gif',

    models_diff_weights_test_ts_outputs_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/test_ts_outputs.gif',


    # Invertion
    # models_diff_weights_invert=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/invert_base.csv',

    # models_diff_weights_invert_interp=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/invert_base.csv',

    models_diff_weights_invert_predictions=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ensemble-{ensemble}/invert_predictions.csv',

    # models_diff_weights_invert_interp_predictions=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/invert_predictions.csv',

    models_diff_weights_invert_loss_history=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ensemble-{ensemble}/invert_history.csv',

    models_diff_weights_invert_saved_weights=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/ensemble-{ensemble}/invert-weights.h5',

    # models_diff_weights_invert_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/invert.gif',
    # models_diff_weights_invert_snake_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/invert_snake.gif',
    # models_diff_weights_invert_ts_outputs_interp_gif=_prefix+'/models/diff_weights/interp-{interp}/method-{method}/activation-{activation}/state-{state}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/invert_ts_outputs.gif',



    #################### OPERATOR ####################
    operators=_prefix+'/operators/method-{method}/state-{state}/mu-{mu}/sigma-{sigma}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base.csv',
    operators_multi=_prefix+'/operators/method-{method}/state-{state}/mu-{mu}/sigma-{sigma}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base_multi.csv',
    operators_saved_weights=_prefix+'/operators/method-{method}/state-{state}/mu-{mu}/sigma-{sigma}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/loss-{loss}/nb_plays#-{__nb_plays__}/weights.h5',
    operators_prediction=_prefix+'/operators/method-{method}/state-{state}/mu-{mu}/sigma-{sigma}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/loss-{loss}/nb_plays#-{__nb_plays__}/prediction.csv',
    operators_loss_history=_prefix+'/operators/method-{method}/state-{state}/mu-{mu}/sigma-{sigma}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/loss-{loss}/history.csv',

    #################### Markov Chain ####################
    # models_mc=_prefix+'/models/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base.csv',
    models_diff_weights_mc=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/base.csv',
    models_diff_weights_mc_predictions=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/predictions.csv',
    models_diff_weights_mc_saved_weights=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/weights.h5',
    models_diff_weights_mc_loss_history=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/loss-{loss}/history.csv',

    # data from stock model
    models_diff_weights_mc_stock_model=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-10000/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/mu-{mu}-sigma-{sigma}-points-{points}.csv',
    models_diff_weights_mc_stock_model_predictions=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble-{ensemble}/loss-{loss}/predictions-batch_size-{batch_size}.csv',
    models_diff_weights_mc_stock_model_trends=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble-{ensemble}/loss-{loss}/trends-batch_size-{batch_size}.csv',
    models_diff_weights_mc_stock_model_trends_list=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble-{ensemble}/loss-{loss}/trends-list-batch_size-{batch_size}.csv',
    models_diff_weights_mc_stock_model_saved_weights=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble-{ensemble}/loss-{loss}weights-batch_size-{batch_size}.h5',
    models_diff_weights_mc_stock_model_loss_history=_prefix+'/models/diff_weights/method-{method}/activation-{activation}/state-{state}/markov_chain/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/input_dim-{input_dim}/predictions-mu-{mu}-sigma-{sigma}-points-{points}/activation#-{__activation__}/state#-{__state__}/units#-{__units__}/nb_plays#-{__nb_plays__}/ensemble-{ensemble}/loss-{loss}/history-batch_size-{batch_size}.csv',


    ################################################################################
    lstm_prediction=_prefix+'/lstm/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/predictions.csv',
    lstm_loss=_prefix+'/lstm/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/loss.csv',
    lstm_loss_file=_prefix+'/lstm/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/mse-loss-lr-{learning_rate}.csv',
    lstm_weights=_prefix+'/lstm/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/model-weights.h5',

    lstm_diff_weights_prediction=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/predictions.csv',
    lstm_diff_weights_loss=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/loss.csv',
    lstm_diff_weights_loss_file=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/history-lr-{learning_rate}.csv',
    lstm_diff_weights_weights=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/units#-{__units__}/ensemble-{ensemble}/loss-{loss}/model-weights.h5',

    lstm_diff_weights_mc_stock_model_prediction=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/markov_chain/units#-{__units__}/activation#-{__activation__}/loss-{loss}/predictions.csv',
    lstm_diff_weights_mc_stock_model_loss=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/markov_chain/units#-{__units__}/activation#-{__activation__}/loss-{loss}/loss.csv',
    lstm_diff_weights_mc_stock_model_loss_file=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/markov_chain/units#-{__units__}/activation#-{__activation__}/loss-{loss}/mle-loss-lr-{learning_rate}.csv',
    lstm_diff_weights_mc_stock_model_weights=_prefix+'/lstm/diff_weights/method-{method}/activation-{activation}/state-{state}/input_dim-{input_dim}/mu-{mu}/sigma-{sigma}/units-{units}/nb_plays-{nb_plays}/points-{points}/markov_chain/units#-{__units__}/activation#-{__activation__}/loss-{loss}/model-weights.h5',


)
