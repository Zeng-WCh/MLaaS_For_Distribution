from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired
from wtforms import FileField, FloatField, IntegerField, StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Optional


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[
                                     DataRequired(), EqualTo('password')])
    submit = SubmitField('注册')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('登录')


class UserModifyForm(FlaskForm):
    username = StringField('Username', validators=[Optional()])
    email = StringField('Email', validators=[Optional(), Email()])
    submit = SubmitField('修改')


class PasswordModifyForm(FlaskForm):
    original_password = PasswordField(
        'Original Password', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[
                                     DataRequired(), EqualTo('password')])
    submit = SubmitField('修改密码')


class ModelSelectionForm(FlaskForm):
    model = SelectField('Select a Model', choices=[])
    submit = SubmitField('Continue')

    def __init__(self, choices=[], *args, **kwargs):
        super(ModelSelectionForm, self).__init__(*args, **kwargs)
        self.model.choices = choices


class SVMConfigForm(FlaskForm):
    lr = FloatField('Learning Rate', validators=[DataRequired()])
    iterations = IntegerField('Iterations', validators=[DataRequired()])
    batch_size = IntegerField('Batch Size', validators=[DataRequired()])
    input_dim = IntegerField('Input Dimension', validators=[DataRequired()])
    output_dim = IntegerField('Output Dimension', validators=[DataRequired()])
    data = FileField('Data File', validators=[FileRequired()])
    datatype = SelectField('Data Type', choices=['json', ])
    submit = SubmitField('提交')


class FullConnectConfigForm(FlaskForm):
    lr = FloatField('Learning Rate', validators=[DataRequired()])
    iterations = IntegerField('Iterations', validators=[DataRequired()])
    batch_size = IntegerField('Batch Size', validators=[DataRequired()])
    input_dim = IntegerField('Input Dimension', validators=[DataRequired()])
    n_hidden_1 = IntegerField(
        'Hidden Layer 1 Dimension', validators=[DataRequired()])
    n_hidden_2 = IntegerField(
        'Hidden Layer 2 Dimension', validators=[DataRequired()])
    output_dim = IntegerField('Output Dimension', validators=[DataRequired()])
    data = FileField('Data File', validators=[])
    datatype = SelectField('Data Type', choices=['MNIST', 'CIFAR10'])
    submit = SubmitField('提交')


class LRConfigForm(FlaskForm):
    lr = FloatField('Learning Rate', validators=[DataRequired()])
    iterations = IntegerField('Iterations', validators=[DataRequired()])
    batch_size = IntegerField('Batch Size', validators=[DataRequired()])
    input_dim = IntegerField('Input Dimension', validators=[DataRequired()])
    output_dim = IntegerField('Output Dimension', validators=[DataRequired()])
    data = FileField('Data File', validators=[FileRequired()])
    datatype = SelectField('Data Type', choices=['json', 'self_defined'])
    submit = SubmitField('提交')


class CNNConfigForm(FlaskForm):
    lr = FloatField('Learning Rate', validators=[DataRequired()])
    iterations = IntegerField('Iterations', validators=[DataRequired()])
    batch_size = IntegerField('Batch Size', validators=[DataRequired()])
    input_dim = IntegerField('Input Dimension', validators=[DataRequired()])
    output_dim = IntegerField('Output Dimension', validators=[DataRequired()])
    data = FileField('Data File', validators=[FileRequired()])
    datatype = SelectField('Data Type', choices=['json', 'self_defined'])
    submit = SubmitField('提交')
