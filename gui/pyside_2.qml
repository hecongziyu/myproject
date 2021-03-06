import QtQuick 2.0
import QtQuick.Layouts 1.12
import QtQuick.Controls 2.12
import QtQuick.Window 2.12
import QtQuick.Controls.Material 2.12

ApplicationWindow {
    id: page
    width: 800
    height: 400
    visible: true

    GridLayout {
        id: grid
        columns: 2
        rows: 3

        ColumnLayout {
            spacing: 2
            Layout.preferredWidth: 400

            Text {
                id: leftlabel
                Layout.alignment: Qt.AlignHCenter
                color: "white"
                font.pointSize: 16
                text: "Qt for Python"
                Layout.preferredHeight: 100
                Material.accent: Material.Green
            }

            RadioButton {
                id: italic
                text: "Italic"
                onToggled: {
                    leftlabel.font.italic = con.getItalic(italic.text)
                    leftlabel.font.bold = con.getBold(italic.text)
                    leftlabel.font.underline = con.getUnderline(italic.text)

                }
            }
            RadioButton {
                id: bold
                text: "Bold"
                onToggled: {
                    leftlabel.font.italic = con.getItalic(bold.text)
                    leftlabel.font.bold = con.getBold(bold.text)
                    leftlabel.font.underline = con.getUnderline(bold.text)
                }
            }
            RadioButton {
                id: underline
                text: "Underline"
                onToggled: {
                    leftlabel.font.italic = con.getItalic(underline.text)
                    leftlabel.font.bold = con.getBold(underline.text)
                    leftlabel.font.underline = con.getUnderline(underline.text)
                }
            }
            RadioButton {
                id: noneradio
                text: "None"
                checked: true
                onToggled: {
                    leftlabel.font.italic = con.getItalic(noneradio.text)
                    leftlabel.font.bold = con.getBold(noneradio.text)
                    leftlabel.font.underline = con.getUnderline(noneradio.text)
                }
            }
        }

        ColumnLayout {
            id: rightcolumn
            spacing: 2
            Layout.columnSpan: 1
            Layout.preferredWidth: 400
            Layout.preferredHeight: 400
            Layout.fillWidth: true

            RowLayout {
                Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter


                Button {
                    id: red
                    text: "Red"
                    highlighted: true
                    Material.accent: Material.Red
                    onClicked: {
                        leftlabel.color = con.getColor(red.text)
                    }
                }
                Button {
                    id: green
                    text: "Green"
                    highlighted: true
                    Material.accent: Material.Green
                    onClicked: {
                        leftlabel.color = con.getColor(green.text)
                    }
                }
                Button {
                    id: blue
                    text: "Blue"
                    highlighted: true
                    Material.accent: Material.Blue
                    onClicked: {
                        leftlabel.color = con.getColor(blue.text)
                    }
                }
                Button {
                    id: nonebutton
                    text: "None"
                    highlighted: true
                    Material.accent: Material.BlueGrey
                    onClicked: {
                        leftlabel.color = con.getColor(nonebutton.text)
                    }
                }
            }
            RowLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
                Text {
                    id: rightlabel
                    color: "white"
                    text: "Font size"
                    Material.accent: Material.White
                }
                Slider {
                    width: rightcolumn.width*0.6
                    Layout.alignment: Qt.AlignRight
                    id: slider
                    value: 0.5
                    onValueChanged: {
                        leftlabel.font.pointSize = con.getSize(value)
                    }
                }
            }
        }
    }
}
