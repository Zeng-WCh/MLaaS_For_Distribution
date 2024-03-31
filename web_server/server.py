import os
import requests

from config import SQL_URI, SECRET_KEY, API_SERVER
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, url_for, flash, redirect, request, session, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from forms import RegistrationForm, LoginForm, UserModifyForm, PasswordModifyForm, ModelSelectionForm
from forms import SVMConfigForm, FullConnectConfigForm, LRConfigForm, CNNConfigForm
from db_model import UserDBLine, TrainDBLine, db
from utils import init_logger, reset_logger_level


app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = SQL_URI
app.config['UPLOAD_FOLDER'] = './uploads'
login_manager = LoginManager()
CSRFProtect(app)

db.init_app(app)
login_manager.init_app(app)

logger = init_logger('web_server', log_file='web_server.log', log_stream=None)


class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username


def get_model_info():
    # GET on API server
    # Send a GET request to the API server
    request_url = f'{API_SERVER}/api/models'
    logger.info(f'Send GET request to API server: {request_url}')
    try:
        response = requests.get(request_url)
    except Exception as e:
        logger.error(f'Failed to get model list from API server: {e}')
    if response.status_code != 200:
        logger.error(f'Failed to get model list from API server!')
        return {}
    else:
        models = response.json()
        logger.debug(f'Get model list from API server: {models}')
        return models


def get_model_params(model_name):
    # GET on API server
    # Send a GET request to the API server
    request_url = f'{API_SERVER}/api/required_params'
    logger.info(f'Send GET request to API server: {request_url}')
    try:
        response = requests.get(request_url, params={'model': model_name})
    except Exception as e:
        logger.error(f'Failed to get model param from API server: {e}')
    if response.status_code != 200:
        logger.error(f'Failed to get model param from API server!')
        return {}
    else:
        models = response.json()
        logger.debug(f'Get model param from API server: {models}')
        return models


models = get_model_info()
model_params = {k: get_model_params(k) for k in models.keys()}

logger.info(f'Get model list: {models}')
logger.info(f'Get model params: {model_params}')


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        logger.debug(f'Get register form: {form}')
        hashed_password = generate_password_hash(form.password.data)
        user = UserDBLine(username=form.username.data,
                          email=form.email.data, password=hashed_password)
        # Check if email has already existed
        if UserDBLine.query.filter_by(email=form.email.data).first():
            flash('This email has already been registered, please log in!', 'alert')
            return redirect(url_for('login'))
        elif UserDBLine.query.filter_by(username=form.username.data).first():
            flash('This username has already been registered, please log in!', 'alert')
            return redirect(url_for('login'))
        db.session.add(user)
        db.session.commit()
        # Get User id
        user = UserDBLine.query.filter_by(username=form.username.data).first()
        user_id = user.id
        flash('Your account has been created! You are now able to log in', 'alert')
        if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{user_id}'):
            os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{user_id}')
        return redirect(url_for('login'))
    else:
        if (len(form.errors)) != 0:
            error = form.errors.popitem()[1][0]
            logger.error(
                f'Get error while handling register form: {form.errors}')
            # flash the first error to the user
            flash(error, 'alert')
        else:
            pass
    # session.pop("_flashes", None)
    return render_template('register.html', title='Register', form=form)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Check password and user email
        user = UserDBLine.query.filter_by(email=form.email.data).first()
        if user is not None:
            logger.debug(
                f'Get user: {user.username}, {user.email}, {user.password}')
        if user is None:
            flash('Email is unregistered!', 'alert')
            return redirect(url_for('register'))
        elif user is not None and check_password_hash(user.password, form.password.data):
            logger.info(
                f'User {user.username} has logged in at {datetime.now()}!')
            curr_user = User(user.id, user.username)
            login_user(curr_user)
            flash('You have been logged in!', 'alert')
            return redirect(request.args.get('next') or url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password!', 'alert')
    # session.pop("_flashes", None)
    return render_template('login.html', title='Login', form=form)


@app.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    logger.info(
        f'User {username} has logged out at {datetime.now()}!')
    return redirect(url_for('login'))


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
@login_required
def home():
    session.pop("_flashes", None)
    if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}'):
        os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}')
    return render_template('home.html', username=current_user.username)


@app.route("/history", methods=['GET', 'POST'])
@login_required
def history():
    if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}'):
        os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}')

    query = db.session.query(TrainDBLine).join(
        UserDBLine, TrainDBLine.created_by == UserDBLine.id).filter(UserDBLine.id == current_user.id).all()

    # to a list of dictionary
    data = []
    for record in query:
        data.append({
            'model_id': record.id,
            'submit_date': record.submit_date.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None),
            'model_choice': record.model_choice,
            'training_status': record.training_status,
            'completed_at': record.completed_at if record.completed_at is None else record.completed_at.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
        })

    logger.debug(f'Get history data from db: {data}')

    return render_template('history.html', username=current_user.username, data=data)


@app.route("/setting", methods=['GET', 'POST'])
@login_required
def setting():
    if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}'):
        os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}')

    form = UserModifyForm()
    if form.validate_on_submit():
        # if email is not empty, update email
        # Check if email has already existed
        email = None
        if form.email.data is not None and form.email.data != '':
            if UserDBLine.query.filter_by(email=email).first():
                flash('This email has already existed, please try another one!', 'alert')
                return redirect(url_for('setting'))
            else:
                email = form.email.data

        username = None
        if form.username.data is not None and form.username.data != '':
            if UserDBLine.query.filter_by(username=username).first():
                flash(
                    'This username has already existed, please try another one!', 'alert')
                return redirect(url_for('setting'))
            else:
                username = form.username.data

        logger.debug(f'Get updated email: {email}')
        logger.debug(f'Get updated username: {username}')
        # Update db
        user = UserDBLine.query.filter_by(id=current_user.id).first()
        if email is not None:
            user.email = email
        if username is not None:
            user.username = username
        db.session.commit()
        flash('Your account has been updated!', 'alert')
        return redirect(url_for('setting'))

    return render_template('setting.html', username=current_user.username, form=form)


@app.route("/password", methods=['GET', 'POST'])
@login_required
def password_modify():
    if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}'):
        os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}')

    form = PasswordModifyForm()

    if form.validate_on_submit():
        original_password = form.original_password.data
        # Check if original password is correct
        user = UserDBLine.query.filter_by(id=current_user.id).first()
        if not check_password_hash(user.password, original_password):
            flash('Original Password is incorrect!', 'alert')
            return redirect(url_for('password_modify'))

        hashed_password = generate_password_hash(form.password.data)
        # Update db
        user.password = hashed_password
        db.session.commit()
        flash('Password Update Successfully!', 'alert')
        return redirect(url_for('password_modify'))

    return render_template('password.html', username=current_user.username, form=form)


@app.route("/training", methods=['GET', 'POST'])
@login_required
def training():
    if not os.path.exists(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}'):
        os.mkdir(f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}')

    global models
    model_select_form = ModelSelectionForm(
        ['-'] + [f'{v}({k})' for k, v in models.items()])
    if model_select_form.validate_on_submit():
        logger.debug(model_select_form.model.data)
        if model_select_form.model.data == '-':
            pass
        else:
            model_name = model_select_form.model.data.split(
                '(')[-1].split(')')[0]
            logger.debug(model_name)
            logger.info(
                f'User {current_user.username} has selected model {model_name}')
            # Use the model name as the request argument
            if model_name == 'SVM':
                return redirect(url_for('svm'))
            elif model_name == 'LR':
                return redirect(url_for('lr'))
            elif model_name == 'CNN':
                return redirect(url_for('cnn'))
            elif model_name == 'FullConnect':
                return redirect(url_for('fullconnect'))
    # print(model)
    # else:
    #     logger.error(model_select_form.errors)

    return render_template('training.html', form=model_select_form, username=current_user.username)


@app.route("/models/svm", methods=['GET', 'POST'])
@login_required
def svm():
    global model_params
    model_name = 'SVM'
    params = model_params[model_name]
    logger.debug(f'{type(params)}')
    logger.debug(f'Get model params: {params}')
    form = SVMConfigForm()
    if form.validate_on_submit():
        logger.info('Success!')
        # Get file
        file = form.data.data
        logger.info(f'Get file: {file}')
        filename = secure_filename(file.filename)
        file_path = os.path.abspath(
            f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}/{filename}')
        file.save(file_path)

        model_export_name = f'{current_user.id}_{model_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pt'

        payload = {
            'id': current_user.id,
            'model': model_name,
            'file': file_path,
            'filetype': form.datatype.data,
            'model_export_name': model_export_name,
            'params': {
                'lr': form.lr.data,
                'iterations': form.iterations.data,
                'batch_size': form.batch_size.data,
                'input_dim': form.input_dim.data,
                'output_dim': form.output_dim.data
            }
        }

        # Send a POST request to the API server
        request_url = f'{API_SERVER}/api/start'
        logger.info(f'Send POST request to API server: {request_url}')
        try:
            response__ = requests.post(request_url, json=payload)
        except Exception as e:
            logger.error(f'Failed to send request to API server: {e}')
        else:
            if response__.status_code != 200:
                logger.error(f'Failed to send request to API server!')
                flash('Failed to send request to API server!', 'alert')
                return redirect(url_for('svm'))
            else:
                logger.info(f'Successfully send request to API server!')
                flash('Successfully Submit!', 'alert')
                return redirect(url_for('history'))

    return render_template('svm.html', form=form, username=current_user.username, model=model_name)


@app.route("/models/fullconnect", methods=['GET', 'POST'])
@login_required
def fullconnect():
    global model_params
    model_name = 'FullConnect'
    params = model_params[model_name]
    logger.debug(f'{type(params)}')
    logger.debug(f'Get model params: {params}')
    form = FullConnectConfigForm()
    if form.validate_on_submit():
        logger.info('Success!')
        # Get file
        file = form.data.data
        if file:
            logger.info(f'Get file: {file}')
            filename = secure_filename(file.filename)
            file_path = os.path.abspath(
                f'{app.config["UPLOAD_FOLDER"]}/{current_user.id}/{filename}')
            file.save(file_path)

        model_export_name = f'{current_user.id}_{model_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.pt'

        payload = {
            'id': current_user.id,
            'model': model_name,
            'file': file_path if file else "",
            'filetype': form.datatype.data,
            'model_export_name': model_export_name,
            'params': {
                'lr': form.lr.data,
                'iterations': form.iterations.data,
                'batch_size': form.batch_size.data,
                'input_dim': form.input_dim.data,
                'n_hidden_1': form.n_hidden_1.data,
                'n_hidden_2': form.n_hidden_2.data,
                'output_dim': form.output_dim.data
            }
        }

        # Send a POST request to the API server
        request_url = f'{API_SERVER}/api/start'
        logger.info(f'Send POST request to API server: {request_url}')
        try:
            response = requests.post(request_url, json=payload)
        except Exception as e:
            logger.error(f'Failed to send request to API server: {e}')
        if response.status_code != 200:
            logger.error(f'Failed to send request to API server!')
            flash('Failed to send request to API server!', 'alert')
            return redirect(url_for('fullconnect'))
        else:
            logger.info(f'Successfully send request to API server!')
            flash('Successfully Submit!', 'alert')
            return redirect(url_for('history'))

    return render_template('fullconnect.html', form=form, username=current_user.username, model=model_name)


@app.route("/models/lr", methods=['GET', 'POST'])
@login_required
def lr():
    global model_params
    model_name = 'LR'
    params = model_params[model_name]
    logger.debug(f'{type(params)}')
    logger.debug(f'Get model params: {params}')
    form = LRConfigForm()
    if form.validate_on_submit():
        logger.info('Success!')

    return render_template('lr.html', form=form, username=current_user.username, model=model_name)


@app.route("/models/cnn", methods=['GET', 'POST'])
@login_required
def cnn():
    global model_params
    model_name = 'CNN'
    params = model_params[model_name]
    logger.debug(f'{type(params)}')
    logger.debug(f'Get model params: {params}')
    form = CNNConfigForm()
    if form.validate_on_submit():
        logger.info('Success!')

    return render_template('cnn.html', form=form, username=current_user.username, model=model_name)


@app.route("/download/<model_id>", methods=['GET'])
@login_required
def download(model_id):
    model = TrainDBLine.query.filter_by(id=model_id).first()
    model_path = model.model_location
    logger.debug(f'Get model path: {model_path}')
    if model is None:
        flash('Model not found!', 'alert')
        return redirect(url_for('history'))
    else:
        if model.training_status == "completed":
            # make a
            return send_file(model_path, as_attachment=True, download_name=model.model_name)
            # return redirect(url_for('static', filename=f'export/{model.model_export_name}'))
        else:
            flash('Model is not completed yet!', 'alert')
            return redirect(url_for('history'))


@login_manager.unauthorized_handler
def unauthorized_callback():
    # session.pop("_flashes", None)
    return redirect(url_for('login', next=request.endpoint))


@login_manager.user_loader
def load_users(user_id):
    user = db.session.get(UserDBLine, user_id)
    if user is not None:
        return User(id=user.id, username=user.username)
    else:
        return None


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0')
