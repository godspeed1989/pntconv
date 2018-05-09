class colors(object):
    PINK = '\033[95m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    #
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    @classmethod
    def info(cls, message):
        return cls.bold(cls.colorize(message, cls.BLUE))
    @classmethod
    def warning(cls, message):
        return cls.bold(cls.colorize(message, cls.YELLOW))
    @classmethod
    def error(cls, message):
        return cls.bold(cls.colorize(message, cls.RED))
    @classmethod
    def success(cls, message):
        return cls.bold(cls.colorize(message, cls.GREEN))
    @classmethod
    def header(cls, message):
        return cls.bold(cls.colorize(message, cls.PINK))
    @classmethod
    def colorize(cls, message, color):
        return "{pre}{msg}{pos}".format(pre=color, msg=message, pos=cls.ENDC)
    @classmethod
    def bold(cls, message):
        return "{pre}{msg}{pos}".format(pre=cls.BOLD, msg=message, pos=cls.ENDC)
    @classmethod
    def underline(cls, message):
        return "{pre}{msg}{pos}".format(pre=cls.UNDERLINE, msg=message, pos=cls.ENDC)


if __name__ == "__main__":
    print(colors.warning("warning"))
    print(colors.info("infomation"))
    print(colors.error("error"))
    print(colors.success("success"))
    print(colors.header("header"))
    print(colors.underline(colors.success("Bold Green Underline")))
